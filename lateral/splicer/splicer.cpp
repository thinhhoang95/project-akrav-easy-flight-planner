#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <tuple>
#include <iostream>
#include <numeric>
#include <chrono>

namespace py = pybind11;

// ----- Utility: hash for std::pair<int,int> -----
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>()(p.first);
        auto h2 = std::hash<int>()(p.second);
        return h1 ^ (h2 << 1);
    }
};

// ----- Graph and Node definitions -----
// (Using integer node identifiers rather than strings.)
struct Node {
    double lat;
    double lon;
    std::vector<int> successors;
    std::vector<int> predecessors;
};

class Graph {
public:
    // Map from node id to node data.
    std::unordered_map<int, Node> nodes;
    // edges[from][to] = distance
    std::unordered_map<int, std::unordered_map<int, double>> edges;
    // Cache for shortest paths: key is (source, target)
    mutable std::unordered_map<std::pair<int,int>, std::vector<int>, PairHash> shortest_path_cache;

    Graph() {}

    void add_node(int id, double lat, double lon) {
        nodes[id] = Node{lat, lon, {}, {}};
        // When the graph changes, clear the cache.
        shortest_path_cache.clear();
    }

    void add_edge(int from, int to, double distance) {
        edges[from][to] = distance;
        if (nodes.find(from) != nodes.end()) {
            nodes[from].successors.push_back(to);
        }
        if (nodes.find(to) != nodes.end()) {
            nodes[to].predecessors.push_back(from);
        }
        shortest_path_cache.clear();
    }

    bool has_edge(int from, int to) const {
        auto it = edges.find(from);
        if (it != edges.end()) {
            return it->second.find(to) != it->second.end();
        }
        return false;
    }

    double get_edge_distance(int from, int to) const {
        if(has_edge(from, to))
            return edges.at(from).at(to);
        return 1.0; // default distance if not specified
    }

    std::vector<int> get_successors(int node) const {
        if(nodes.find(node) != nodes.end())
            return nodes.at(node).successors;
        return {};
    }

    std::vector<int> get_predecessors(int node) const {
        if(nodes.find(node) != nodes.end())
            return nodes.at(node).predecessors;
        return {};
    }
};

// ----- Utility functions -----
// Euclidean distance between two points.
double euclidean_distance(double lat1, double lon1, double lat2, double lon2) {
    return std::sqrt((lat1 - lat2) * (lat1 - lat2) +
                     (lon1 - lon2) * (lon1 - lon2));
}

// Check if two segments (defined by endpoints) are in the same general direction.
bool is_same_direction(const Node &origin, const Node &dest,
                       const Node &node, const Node &neighbor) {
    double v1x = dest.lat - origin.lat;
    double v1y = dest.lon - origin.lon;
    double v2x = neighbor.lat - node.lat;
    double v2y = neighbor.lon - node.lon;
    double dot = v1x * v2x + v1y * v2y;
    return dot > 0;
}

bool is_same_direction_by_wp_id(const Graph &G, int origin_id, int dest_id,
                                int node_id, int neighbor_id) {
    const Node &origin = G.nodes.at(origin_id);
    const Node &dest = G.nodes.at(dest_id);
    const Node &node = G.nodes.at(node_id);
    const Node &neighbor = G.nodes.at(neighbor_id);
    return is_same_direction(origin, dest, node, neighbor);
}

// ----- BFS-based admissible pivot nodes search -----
struct BFSResult {
    std::unordered_map<int, std::pair<int, double>> distances; // node -> (depth, total_distance)
    std::vector<int> admissible_nodes;
};

BFSResult find_admissible_pivot_nodes_with_heuristics(const Graph &G,
                                                      int node_from,
                                                      const std::unordered_set<int> &nodes_to_exclude,
                                                      int max_depth, bool prevent_backtracking,
                                                      int origin, int dest,
                                                      const std::string &direction,
                                                      int branching_factor) {
    BFSResult result;
    result.distances[node_from] = {0, 0.0};
    std::vector<int> buff = {node_from};
    std::unordered_set<int> admissible_set;

    while(!buff.empty()){
        std::vector<int> next_level;
        int current_depth = result.distances[buff[0]].first;
        if(current_depth >= max_depth)
            break;

        for(const auto &node : buff) {
            std::vector<int> neighbors;
            if(direction == "forward")
                neighbors = G.get_successors(node);
            else
                neighbors = G.get_predecessors(node);

            for(const auto &neighbor : neighbors) {
                if(nodes_to_exclude.find(neighbor) != nodes_to_exclude.end())
                    continue;
                if(prevent_backtracking) {
                    bool backtrack;
                    if(direction == "forward")
                        backtrack = !is_same_direction_by_wp_id(G, origin, dest, node, neighbor);
                    else
                        backtrack = !is_same_direction_by_wp_id(G, dest, origin, node, neighbor);
                    if(backtrack)
                        continue;
                }
                double edge_distance = 1.0;
                if(direction == "forward") {
                    if(G.has_edge(node, neighbor))
                        edge_distance = G.get_edge_distance(node, neighbor);
                } else {
                    if(G.has_edge(neighbor, node))
                        edge_distance = G.get_edge_distance(neighbor, node);
                }
                double new_distance = result.distances[node].second + edge_distance;
                int new_depth = current_depth + 1;
                if(result.distances.find(neighbor) == result.distances.end() ||
                   new_depth < result.distances[neighbor].first ||
                   (new_depth == result.distances[neighbor].first &&
                    new_distance < result.distances[neighbor].second)) {
                    result.distances[neighbor] = {new_depth, new_distance};
                    next_level.push_back(neighbor);
                    admissible_set.insert(neighbor);
                }
            }
        }
        std::sort(next_level.begin(), next_level.end());
        next_level.erase(std::unique(next_level.begin(), next_level.end()), next_level.end());

        if(dest != -1) {
            int target = (direction == "forward") ? dest : origin;
            std::sort(next_level.begin(), next_level.end(), [&](int n1, int n2) {
                double score1 = result.distances[n1].second +
                    euclidean_distance(G.nodes.at(n1).lat, G.nodes.at(n1).lon,
                                       G.nodes.at(target).lat, G.nodes.at(target).lon);
                double score2 = result.distances[n2].second +
                    euclidean_distance(G.nodes.at(n2).lat, G.nodes.at(n2).lon,
                                       G.nodes.at(target).lat, G.nodes.at(target).lon);
                return score1 < score2;
            });
        } else {
            std::sort(next_level.begin(), next_level.end(), [&](int n1, int n2) {
                return result.distances[n1].second < result.distances[n2].second;
            });
        }

        if(branching_factor > 0 && next_level.size() > static_cast<size_t>(branching_factor))
            next_level.resize(branching_factor);
        buff = next_level;
    }
    result.admissible_nodes.assign(admissible_set.begin(), admissible_set.end());
    return result;
}

// ----- Pivot probability computations -----
struct PivotDistance {
    std::pair<int, double> forward_dist;
    std::pair<int, double> backward_dist;
};

std::pair<std::unordered_map<int, PivotDistance>, std::unordered_map<int, int>>
collapse_pivot_options(const std::unordered_map<int, PivotDistance> &V_distances,
                        double threshold = 1e-6) {
    std::vector<std::pair<int, double>> nodes_with_dist;
    for(const auto &item : V_distances) {
        double total = item.second.forward_dist.second + item.second.backward_dist.second;
        nodes_with_dist.push_back({item.first, total});
    }
    std::sort(nodes_with_dist.begin(), nodes_with_dist.end(),
              [](auto &a, auto &b) { return a.second < b.second; });
    std::unordered_map<int, PivotDistance> collapsed;
    std::unordered_map<int, int> mapping;
    std::vector<int> current_group;
    double current_dist = 0.0;
    bool first = true;
    for(const auto &pair : nodes_with_dist) {
        if(first) {
            current_group.push_back(pair.first);
            current_dist = pair.second;
            first = false;
        } else {
            if(std::fabs(pair.second - current_dist) > threshold) {
                if(!current_group.empty()){
                    int rep = current_group[0];
                    collapsed[rep] = V_distances.at(rep);
                    for(const auto &n : current_group)
                        mapping[n] = rep;
                }
                current_group.clear();
                current_group.push_back(pair.first);
                current_dist = pair.second;
            } else {
                current_group.push_back(pair.first);
            }
        }
    }
    if(!current_group.empty()){
        int rep = current_group[0];
        collapsed[rep] = V_distances.at(rep);
        for(const auto &n : current_group)
            mapping[n] = rep;
    }
    return {collapsed, mapping};
}

std::vector<double> compute_pivot_probabilities(const std::unordered_map<int, PivotDistance> &pivots,
                                                  double mu = 1.0) {
    std::vector<double> totals;
    for(const auto &item : pivots) {
        double total = item.second.forward_dist.second + item.second.backward_dist.second;
        totals.push_back(total);
    }
    if(totals.empty())
        return {};
    double min_total = *std::min_element(totals.begin(), totals.end());
    std::vector<double> exp_totals;
    for(double t : totals)
        exp_totals.push_back(std::exp(-(t - min_total) / mu));
    double sum = std::accumulate(exp_totals.begin(), exp_totals.end(), 0.0);
    std::vector<double> probabilities;
    for(double val : exp_totals)
        probabilities.push_back(val / sum);
    return probabilities;
}

std::tuple<int, double, double> sample_pivot(const std::unordered_map<int, PivotDistance> &pivots,
                                               const std::vector<double> &probabilities,
                                               double mu = 1.0,
                                               int old_pivot = -1,
                                               const std::unordered_map<int, int> *node_to_collapsed = nullptr) {
    std::vector<int> keys;
    for(const auto &item : pivots)
        keys.push_back(item.first);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    int sampled_index = dist(gen);
    int sampled_pivot = keys[sampled_index];
    double sampled_prob = probabilities[sampled_index];
    double old_pivot_prob = 0.0;
    if(old_pivot != -1) {
        auto it = std::find(keys.begin(), keys.end(), old_pivot);
        if(it != keys.end()) {
            int idx = std::distance(keys.begin(), it);
            old_pivot_prob = probabilities[idx];
        } else if(node_to_collapsed) {
            auto map_it = node_to_collapsed->find(old_pivot);
            if(map_it != node_to_collapsed->end()){
                int rep = map_it->second;
                auto rep_it = std::find(keys.begin(), keys.end(), rep);
                if(rep_it != keys.end()){
                    int idx = std::distance(keys.begin(), rep_it);
                    old_pivot_prob = probabilities[idx];
                }
            }
        }
    }
    return {sampled_pivot, sampled_prob, old_pivot_prob};
}

// ----- Shortest path using Dijkstra's algorithm with caching -----
// Since the graph is static, we cache each computed shortest path.
std::vector<int> get_shortest_path(Graph &G, int source, int target) {
    std::pair<int, int> key = {source, target};
    if(G.shortest_path_cache.find(key) != G.shortest_path_cache.end())
        return G.shortest_path_cache[key];

    std::unordered_map<int, double> dist;
    std::unordered_map<int, int> prev;
    for(const auto &pair : G.nodes)
        dist[pair.first] = std::numeric_limits<double>::infinity();
    dist[source] = 0.0;
    using Pair = std::pair<double, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;
    pq.push({0.0, source});
    while(!pq.empty()){
        auto [d, u] = pq.top();
        pq.pop();
        if(d > dist[u])
            continue;
        if(u == target)
            break;
        for(const auto &v : G.get_successors(u)) {
            if(G.has_edge(u, v)){
                double alt = d + G.get_edge_distance(u, v);
                if(alt < dist[v]){
                    dist[v] = alt;
                    prev[v] = u;
                    pq.push({alt, v});
                }
            }
        }
    }
    std::vector<int> path;
    int cur = target;
    if(prev.find(cur) == prev.end() && cur != source) {
        G.shortest_path_cache[key] = {}; // No path found.
        return {};
    }
    while(cur != source) {
        path.push_back(cur);
        cur = prev[cur];
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end());
    G.shortest_path_cache[key] = path;
    return path;
}

// ----- Replace a segment of the route -----
std::tuple<std::vector<int>, int, int> replace_route_segment(Graph &G,
                                                             const std::vector<int> &route,
                                                             int a, int c, int v) {
    std::vector<int> new_route;
    // Prefix: route[0, a)
    new_route.insert(new_route.end(), route.begin(), route.begin() + a);
    // sp1: shortest path from route[a-1] to v
    std::vector<int> sp1 = get_shortest_path(G, route[a-1], v);
    // sp2: shortest path from v to route[c]
    std::vector<int> sp2 = get_shortest_path(G, v, route[c]);
    if(!sp1.empty())
        new_route.insert(new_route.end(), sp1.begin() + 1, sp1.end());
    if(!sp2.empty())
        new_route.insert(new_route.end(), sp2.begin() + 1, sp2.end());
    // Suffix: route[c+1, end)
    if(c + 1 < static_cast<int>(route.size()))
        new_route.insert(new_route.end(), route.begin() + c + 1, route.end());
    int new_a = a;
    int new_c = static_cast<int>(new_route.size()) - (static_cast<int>(route.size()) - c);
    return {new_route, new_a, new_c};
}

// ----- Evaluate the total distance (cost) of a route -----
double evaluate_route(const Graph &G, const std::vector<int> &route) {
    double total = 0.0;
    for (size_t i = 0; i < route.size() - 1; i++) {
        if(G.has_edge(route[i], route[i+1]))
            total += G.get_edge_distance(route[i], route[i+1]);
    }
    return total;
}

// ----- Metropolis-Hastings acceptance probability -----
double mh_acceptance(double cost_new, double cost_old, double p_forward, double p_backward) {
    double cost_diff = -(cost_new - cost_old);
    double prob_ratio = (p_forward == 0.0 ? 0.0 : p_backward / p_forward);
    double alpha = std::min(1.0, std::exp(cost_diff) * prob_ratio);
    return alpha;
}

// ----- Determine eligible range of nodes to sample -----
// Here, all interior nodes are eligible.
std::pair<int, int> find_eligible_range_of_nodes_to_sample(const std::vector<int> &route) {
    if(route.size() < 3)
        return {-1, -1};
    std::vector<int> eligible;
    for (int i = 1; i < static_cast<int>(route.size()) - 1; i++)
        eligible.push_back(i);
    if(eligible.empty())
        return {-1, -1};
    int longest_start = eligible[0];
    int current_start = eligible[0];
    int longest_length = 1;
    int current_length = 1;
    for (size_t i = 1; i < eligible.size(); i++) {
        if (eligible[i] == eligible[i-1] + 1)
            current_length++;
        else {
            if (current_length > longest_length) {
                longest_length = current_length;
                longest_start = current_start;
            }
            current_start = eligible[i];
            current_length = 1;
        }
    }
    if (current_length > longest_length) {
        longest_length = current_length;
        longest_start = current_start;
    }
    int longest_end = longest_start + longest_length - 1;
    // Adjust boundaries as in the original Python version.
    if(longest_start > 0)
        longest_start++;
    if(longest_end < static_cast<int>(route.size()) - 1)
        longest_end--;
    return {longest_start, longest_end};
}

// ----- The main MCMC step function -----
std::tuple<std::vector<int>, bool> mcmc_step(Graph &G,
                                             const std::vector<int> &route,
                                             double temperature,
                                             int max_depth = 8,
                                             bool prevent_backtracking = true,
                                             bool verbose = false) {
    // 1. Determine eligible range for sampling.
    auto eligible_range = find_eligible_range_of_nodes_to_sample(route);
    if(eligible_range.first == -1)
        return {route, false};
    int a, c;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_a(eligible_range.first, eligible_range.second);
    a = dist_a(gen);
    int c_max = eligible_range.second + 1;
    if(c_max <= a + 1) {
        if(verbose)
            std::cout << "Early rejection: eligible range too narrow." << std::endl;
        return {route, false};
    } else {
        std::uniform_int_distribution<> dist_c(a + 1, c_max);
        c = dist_c(gen);
        if(verbose)
            std::cout << "a = " << a << ", c = " << c << std::endl;
    }
    // 2. Build the set of nodes to exclude.
    std::unordered_set<int> nodes_to_exclude;
    for (int i = 0; i < a; i++)
        nodes_to_exclude.insert(route[i]);
    for (int i = c; i < static_cast<int>(route.size()); i++)
        nodes_to_exclude.insert(route[i]);
    // 3. Find forward and backward admissible pivot nodes.
    BFSResult forward_result = find_admissible_pivot_nodes_with_heuristics(
        G, route[a-1], nodes_to_exclude, max_depth, prevent_backtracking, route[a-1], route[c],
        "forward", 16);
    BFSResult backward_result = find_admissible_pivot_nodes_with_heuristics(
        G, route[c], nodes_to_exclude, max_depth, prevent_backtracking, route[a-1], route[c],
        "backward", 16);
    // 4. Intersection of forward and backward nodes.
    std::unordered_set<int> forward_set(forward_result.admissible_nodes.begin(), forward_result.admissible_nodes.end());
    std::unordered_set<int> backward_set(backward_result.admissible_nodes.begin(), backward_result.admissible_nodes.end());
    std::unordered_set<int> intersection;
    for (const auto &node : forward_set)
        if (backward_set.find(node) != backward_set.end())
            intersection.insert(node);
    // 5. Build a map of pivot nodes with their forward and backward distances.
    std::unordered_map<int, PivotDistance> V_distances;
    for (const auto &v : intersection) {
        PivotDistance pd;
        pd.forward_dist = forward_result.distances[v];
        pd.backward_dist = backward_result.distances[v];
        V_distances[v] = pd;
    }
    if(V_distances.empty()){
        if(verbose)
            std::cout << "No admissible pivot nodes found!" << std::endl;
        return {route, false};
    }
    // 6. Collapse pivot options.
    auto [V_collapsed, node_to_collapsed] = collapse_pivot_options(V_distances);
    // 7. Choose the "old" pivot (with minimum total distance).
    int old_pivot_node = -1;
    double min_total = std::numeric_limits<double>::infinity();
    for (const auto &item : V_collapsed) {
        double total = item.second.forward_dist.second + item.second.backward_dist.second;
        if (total < min_total) {
            min_total = total;
            old_pivot_node = item.first;
        }
    }
    // 8. Compute pivot probabilities and sample a pivot.
    std::vector<double> pivot_probs = compute_pivot_probabilities(V_collapsed, temperature);
    auto [sampled_V, prob_sampled_V, old_pivot_prob] = sample_pivot(V_collapsed, pivot_probs, temperature, old_pivot_node, &node_to_collapsed);
    if(verbose)
        std::cout << "Sampled pivot: " << sampled_V << " with probability " << prob_sampled_V << std::endl;
    // 9. Propose a new route by replacing the segment.
    auto [new_route, new_a, new_c] = replace_route_segment(G, route, a, c, sampled_V);
    if(verbose)
        std::cout << "Route segment replaced." << std::endl;
    // 10. Evaluate costs.
    double cost_new = evaluate_route(G, new_route);
    double cost_old = evaluate_route(G, route);
    if(verbose)
        std::cout << "Cost new: " << cost_new << ", Cost old: " << cost_old << std::endl;
    // 11. Compute acceptance probability and decide.
    double acceptance_prob = mh_acceptance(cost_new, cost_old, prob_sampled_V, old_pivot_prob);
    std::uniform_real_distribution<> dist_uniform(0.0, 1.0);
    bool accepted = (dist_uniform(gen) < acceptance_prob);
    return {accepted ? new_route : route, accepted};
}

// ----- The full MCMC loop function with burn-in and thinning -----
std::tuple<std::vector<std::vector<int>>, std::vector<int>, int, double>
start_mcmc(Graph &G,
           std::vector<int> route,
           double temperature,
           int max_iter,
           int burn_in,
           int thinning,
           int max_depth = 8,
           bool prevent_backtracking = true,
           bool verbose = false) {
    int total_accepted = 0;
    std::vector<std::vector<int>> sampled_routes;
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < max_iter; i++) {
        auto [new_route, accepted] = mcmc_step(G, route, temperature, max_depth, prevent_backtracking, verbose);
        if (accepted) {
            route = new_route;
            total_accepted++;
            // Record sample after burn-in and thinning.
            if (i >= burn_in && ((i - burn_in) % thinning == 0))
                sampled_routes.push_back(route);
        }
        if (verbose)
            std::cout << "Iteration " << i+1 << ", accepted: " << total_accepted << "\r" << std::flush;
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    if (verbose)
        std::cout << "\nTotal time: " << elapsed_seconds.count() << " seconds. Total accepted: " 
                  << total_accepted << "/" << max_iter 
                  << " => acceptance rate: " << static_cast<double>(total_accepted)/max_iter << std::endl;
    double acceptance_rate = static_cast<double>(total_accepted) / max_iter;
    return {sampled_routes, route, total_accepted, acceptance_rate};
}

// ----- Pybind11 module definition -----
PYBIND11_MODULE(splicer, m) {
    m.doc() = "MCMC pivot planner module in C++ using pybind11 with int-based nodes and caching of shortest paths";

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_node", &Graph::add_node, "Add a node",
             py::arg("id"), py::arg("lat"), py::arg("lon"))
        .def("add_edge", &Graph::add_edge, "Add an edge",
             py::arg("from"), py::arg("to"), py::arg("distance"))
        .def("has_edge", &Graph::has_edge, "Check if an edge exists",
             py::arg("from"), py::arg("to"))
        .def("get_edge_distance", &Graph::get_edge_distance, "Get edge distance",
             py::arg("from"), py::arg("to"))
        .def("get_successors", &Graph::get_successors, "Get successors of a node",
             py::arg("node"))
        .def("get_predecessors", &Graph::get_predecessors, "Get predecessors of a node",
             py::arg("node"))
        .def_readwrite("nodes", &Graph::nodes)
        .def_readwrite("edges", &Graph::edges);

    m.def("mcmc_step", &mcmc_step, "Perform one MCMC step",
          py::arg("G"), py::arg("route"), py::arg("temperature"),
          py::arg("max_depth") = 8, py::arg("prevent_backtracking") = true, py::arg("verbose") = false);

    m.def("start_mcmc", &start_mcmc, "Run the full MCMC loop with burn-in and thinning",
          py::arg("G"), py::arg("route"), py::arg("temperature"),
          py::arg("max_iter"), py::arg("burn_in"), py::arg("thinning"),
          py::arg("max_depth") = 8, py::arg("prevent_backtracking") = true, py::arg("verbose") = false);
}
