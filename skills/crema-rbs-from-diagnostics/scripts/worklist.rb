#!/usr/bin/env ruby
# frozen_string_literal: true

# Rank unresolved constants from a `crema check` JSONL by cascade weight.
#
#   ruby worklist.rb crema.jsonl [--top N]
#   ruby worklist.rb crema.jsonl --methods PATH
#
# For every `Ruby::UnknownConstant` record the raw count is the number of
# records sharing its `path` (the constant path as written). The cascade
# weight is the number of `Ruby::NoMethod` records whose receiver is a
# class that (transitively) inherits from, or mixes in, that unresolved
# path: `class Foo < Unknown` leaves every method called on Foo and its
# subclasses unresolvable, so one UnknownConstant can own thousands of
# NoMethod records. Sorting by raw count alone buries those roots.
#
# The class graph comes from Prism (bundled with Ruby >= 3.3) over the
# `.rb` files under the `check` entries of `./crema.toml`; if there is no
# crema.toml in the current directory, only the files named in the JSONL
# are parsed. Superclass and mixin references are resolved lexically
# against the classes found; a reference that resolves to nothing is kept
# as written so it can be matched against the JSONL `path`.
#
# Output is TSV: path, unknown_constant, cascade_no_method, total, roots,
# smoke. `roots` lists up to three `file:line` sites where the path
# appears in a superclass / include / extend / prepend position. `smoke`
# is the file with the most cascading NoMethod records: the file to
# re-check first after a patch. A path written with and without a leading
# `::` is one row.
#
# `--methods PATH` prints the cascade of one path instead: the NoMethod
# method names on every class under PATH's roots, with counts. That is the
# list of methods a patch for PATH has to account for, which a grep for
# `PATH.method` cannot see because the calls are made on the subclasses
# with an implicit receiver.

require "json"
require "prism"

class ClassGraph
  Entry = Struct.new(:fqn, :file, :line, :superclass, :mixins)

  attr_reader :entries

  def initialize
    @entries = {} # fqn => Entry (first definition wins for file:line)
    @edges = Hash.new { |h, k| h[k] = [] } # written parent ref => [(child fqn, file, line)]
  end

  def parse_file(path)
    result = Prism.parse_file(path)
    visit(result.value, [], path)
  rescue StandardError => e
    warn "worklist: skip #{path}: #{e.message}"
  end

  # Children whose superclass / mixin was written as `ref` (after lexical
  # resolution failed, i.e. the reference is unresolved in this project).
  # Both sides are compared without a leading `::`: crema keeps it in the
  # JSONL `path` when the source wrote an absolute reference.
  def roots_for(ref)
    @edges[ref.delete_prefix("::")]
  end

  # Transitive descendants of `fqn` by superclass edge, plus classes that
  # mix in `fqn`. Returns the set including `fqn` itself.
  def closure(fqns)
    seen = {}
    queue = fqns.dup
    until queue.empty?
      cur = queue.shift
      next if seen[cur]

      seen[cur] = true
      @children[cur]&.each { |c| queue << c }
    end
    seen.keys
  end

  def finalize!
    @children = Hash.new { |h, k| h[k] = [] }
    @entries.each_value do |e|
      refs = [e.superclass, *e.mixins].compact
      refs.each do |ref|
        resolved = resolve(ref, e.fqn)
        if resolved
          @children[resolved] << e.fqn
        else
          @edges[ref.delete_prefix("::")] << [e.fqn, e.file, e.line]
        end
      end
    end
  end

  private

  # Lexical lookup: `Base` inside `::API::Foo` tries `::API::Foo::Base`,
  # `::API::Base`, `::Base`. An absolute `::X` is looked up as is.
  def resolve(ref, scope)
    return (@entries.key?(ref) ? ref : nil) if ref.start_with?("::")

    parts = scope.split("::").reject(&:empty?)
    parts.size.downto(0) do |n|
      cand = "::" + (parts.first(n) + [ref]).join("::")
      return cand if @entries.key?(cand)
    end
    nil
  end

  def visit(node, nesting, file)
    case node
    when Prism::ClassNode, Prism::ModuleNode
      fqn = qualify(node.constant_path, nesting)
      if fqn
        entry = (@entries[fqn] ||= Entry.new(fqn, file, node.location.start_line, nil, []))
        if node.is_a?(Prism::ClassNode) && node.superclass
          sup = const_source(node.superclass)
          entry.superclass ||= sup if sup
        end
        entry.mixins.concat(mixins_of(node.body))
        node.body&.compact_child_nodes&.each { |c| visit(c, nesting + [fqn], file) }
      end
    when Prism::SingletonClassNode
      # `class << self; include X; end` mixes X into the singleton, whose
      # NoMethod receivers are `singleton(fqn)` — counted under the same
      # entry as the enclosing class.
      @entries[nesting.last]&.mixins&.concat(mixins_of(node.body)) if nesting.last
      node.body&.compact_child_nodes&.each { |c| visit(c, nesting, file) }
    else
      node.compact_child_nodes.each { |c| visit(c, nesting, file) }
    end
  end

  # `include X`, `extend X`, `prepend X` directly in the class body.
  def mixins_of(body)
    return [] unless body

    body.compact_child_nodes.filter_map do |stmt|
      next unless stmt.is_a?(Prism::CallNode) && stmt.receiver.nil?
      next unless %i[include extend prepend].include?(stmt.name)

      stmt.arguments&.arguments&.filter_map { |a| const_source(a) }
    end.flatten
  end

  def const_source(node)
    case node
    when Prism::ConstantReadNode, Prism::ConstantPathNode then node.slice
    end
  end

  def qualify(path_node, nesting)
    name = const_source(path_node) or return nil
    return name if name.start_with?("::")

    prefix = nesting.last || ""
    "#{prefix}::#{name}"
  end
end

def check_dirs
  return nil unless File.exist?("crema.toml")

  toml = File.read("crema.toml")
  m = toml.match(/^\s*check\s*=\s*\[(.*?)\]/m) or return nil
  m[1].scan(/"([^"]+)"/).flatten
end

jsonl = ARGV.shift or abort "usage: worklist.rb crema.jsonl [--top N | --methods PATH]"
top = 40
methods_for = nil
if (i = ARGV.index("--methods"))
  methods_for = ARGV[i + 1] or abort "worklist.rb: --methods expects a constant path"
end
if (i = ARGV.index("--top"))
  top = Integer(ARGV[i + 1] || abort("usage: worklist.rb crema.jsonl [--top N]"), exception: false) ||
        abort("worklist.rb: --top expects an integer, got #{ARGV[i + 1].inspect}")
end

unknown = Hash.new { |h, k| h[k] = [] }
no_method = Hash.new(0)
no_method_names = Hash.new { |h, k| h[k] = Hash.new(0) } # receiver => method_name => count
no_method_files = Hash.new { |h, k| h[k] = Hash.new(0) } # receiver => file => count
files = {}
File.foreach(jsonl) do |line|
  r = JSON.parse(line)
  files[r["file"]] = true if r["file"]
  case r["code"]
  when "Ruby::UnknownConstant" then unknown[r["path"].delete_prefix("::")] << r
  when "Ruby::NoMethod"
    # A union receiver (`::X | nil`) still cascades from each member.
    # `missing_from` lists the expanded members even when `receiver_type`
    # shows an alias name (`::RBS::Types::t`), so prefer it.
    members = r["missing_from"] || r["receiver_type"].to_s.split(" | ")
    members.each do |t|
      no_method[t] += 1
      no_method_names[t][r["method_name"]] += 1
      no_method_files[t][r["file"]] += 1
    end
  end
end

graph = ClassGraph.new
targets = check_dirs&.flat_map { |d| File.directory?(d) ? Dir.glob("#{d}/**/*.rb") : [d] } || files.keys
targets.each { |f| graph.parse_file(f) if File.file?(f) }
graph.finalize!

if methods_for
  fqns = graph.closure(graph.roots_for(methods_for).map(&:first))
  counts = Hash.new(0)
  fqns.each do |f|
    ["singleton(#{f})", f].each { |t| no_method_names[t].each { |m, c| counts[m] += c } }
  end
  puts %w[count method_name].join("\t")
  counts.sort_by { |m, c| [-c, m] }.each { |m, c| puts [c, m].join("\t") }
  exit
end

rows = unknown.map do |path, records|
  roots = graph.roots_for(path)
  fqns = graph.closure(roots.map(&:first))
  cascade = fqns.sum { |f| no_method["singleton(#{f})"] + no_method[f] }
  per_file = Hash.new(0)
  fqns.each do |f|
    ["singleton(#{f})", f].each { |t| no_method_files[t].each { |file, c| per_file[file] += c } }
  end
  smoke = per_file.max_by { |file, c| [c, file] }&.first || ""
  [path, records.size, cascade, records.size + cascade, roots.first(3).map { |_, f, l| "#{f}:#{l}" }.join(","), smoke]
end

puts %w[path unknown_constant cascade_no_method total roots smoke].join("\t")
rows.sort_by { |r| [-r[3], r[0]] }.first(top).each { |r| puts r.join("\t") }
