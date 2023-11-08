#include "mm/stmatch/stmatch_algorithm.hpp"
#include "algorithm/geom_algorithm.hpp"
#include "util/debug.hpp"
#include "util/util.hpp"
#include "io/gps_reader.hpp"
#include "io/mm_writer.hpp"

#include <limits>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;
using namespace FMM::PYTHON;

STMATCHConfig::STMATCHConfig(
  int k_arg, double r_arg, double gps_error_arg,
  double vmax_arg, double factor_arg, double reverse_tolerance_arg):
  k(k_arg), radius(r_arg), gps_error(gps_error_arg),
  vmax(vmax_arg), factor(factor_arg),
  reverse_tolerance(reverse_tolerance_arg) {
};

void STMATCHConfig::print() const {
  SPDLOG_INFO("STMATCHAlgorithmConfig");
  SPDLOG_INFO("k {} radius {} gps_error {} vmax {} factor {}",
              k, radius, gps_error, vmax, factor);
  SPDLOG_INFO("reverse_tolerance {}",reverse_tolerance);
};

STMATCHConfig STMATCHConfig::load_from_xml(
  const boost::property_tree::ptree &xml_data) {
  int k = xml_data.get("config.parameters.k", 8);
  double radius = xml_data.get("config.parameters.r", 300.0);
  double gps_error = xml_data.get("config.parameters.gps_error", 50.0);
  double vmax = xml_data.get("config.parameters.vmax", 80.0);
  double factor = xml_data.get("config.parameters.factor", 1.5);
  double reverse_tolerance =
    xml_data.get("config.parameters.reverse_tolerance", 0.0);
  return STMATCHConfig{k, radius, gps_error, vmax, factor,reverse_tolerance};
};

STMATCHConfig STMATCHConfig::load_from_arg(
  const cxxopts::ParseResult &arg_data) {
  int k = arg_data["candidates"].as<int>();
  double radius = arg_data["radius"].as<double>();
  double gps_error = arg_data["error"].as<double>();
  double vmax = arg_data["vmax"].as<double>();
  double factor = arg_data["factor"].as<double>();
  double reverse_tolerance = arg_data["reverse_tolerance"].as<double>();
  return STMATCHConfig{k, radius, gps_error, vmax, factor, reverse_tolerance};
};

void STMATCHConfig::register_arg(cxxopts::Options &options){
  options.add_options()
    ("k,candidates","Number of candidates",
    cxxopts::value<int>()->default_value("8"))
    ("r,radius","Search radius",
    cxxopts::value<double>()->default_value("300.0"))
    ("e,error","GPS error",
    cxxopts::value<double>()->default_value("50.0"))
    ("vmax","Maximum speed",
    cxxopts::value<double>()->default_value("30.0"))
    ("factor","Scale factor",
    cxxopts::value<double>()->default_value("1.5"))
    ("reverse_tolerance","Ratio of reverse movement allowed",
      cxxopts::value<double>()->default_value("0.0"));
}

void STMATCHConfig::register_help(std::ostringstream &oss){
  oss<<"-k/--candidates (optional) <int>: number of candidates (8)\n";
  oss<<"-r/--radius (optional) <double>: search "
    "radius (network data unit) (300)\n";
  oss<<"-e/--error (optional) <double>: GPS error "
    "(network data unit) (50)\n";
  oss<<"-f/--factor (optional) <double>: scale factor (1.5)\n";
  oss<<"-v/--vmax (optional) <double>: "
    " Maximum speed (unit: network_data_unit/s) (30)\n";
  oss<<"--reverse_tolerance (optional) <double>: proportion "
      "of reverse movement allowed on an edge\n";
};

bool STMATCHConfig::validate() const {
  if (gps_error <= 0 || radius <= 0 || k <= 0 || vmax <= 0 || factor <= 0
      || reverse_tolerance<0) {
    SPDLOG_CRITICAL("Invalid mm parameter k {} r {} gps error {} "
        "vmax {} f {} reverse_tolerance {}",
                    k, radius, gps_error, vmax, factor, reverse_tolerance);
    return false;
  }
  return true;
}

std::vector<PyMatchResult> STMATCH::match_wkt(
  const std::string &wkt, const std::vector<double> &timestamps, const STMATCHConfig &config) {
  LineString line = wkt2linestring(wkt);
  Trajectory traj{0, line, timestamps};
  std::vector<MatchResult> results = match_traj(traj, config);
  std::vector<PyMatchResult> outputs;
  for (MatchResult result:results){
    PyMatchResult output;
    output.id = result.id;
    output.opath = result.opath;
    output.cpath = result.cpath;
    output.mgeom = result.mgeom;
    output.indices = result.indices;
    for (int i = 0; i < result.opt_candidate_path.size(); ++i) {
      const MatchedCandidate &mc = result.opt_candidate_path[i];
      output.candidates.push_back(
        {i,
        mc.c.edge->id,
        graph_.get_node_id(mc.c.edge->source),
        graph_.get_node_id(mc.c.edge->target),
        mc.c.dist,
        mc.c.offset,
        mc.c.edge->length,
        mc.ep,
        mc.tp,
        mc.sp_dist}
        );
      output.pgeom.add_point(mc.c.point);
    }
    outputs.push_back(output);
  }
  return outputs;
};

// Procedure of HMM based map matching algorithm.
std::vector<MatchResult> STMATCH::match_traj(const Trajectory &traj,
                                const STMATCHConfig &config) {
  SPDLOG_DEBUG("Count of points in trajectory {}", traj.geom.get_num_points());
  SPDLOG_DEBUG("Search candidates");
  Traj_Candidates tc = network_.search_tr_cs_knn(
    traj.geom, config.k, config.radius);
  SPDLOG_DEBUG("Trajectory candidate {}", tc);
  if (tc.empty()) return std::vector<MatchResult>{};
  SPDLOG_DEBUG("Generate dummy graph");
  DummyGraph dg(tc, config.reverse_tolerance);
  SPDLOG_DEBUG("Generate composite_graph");
  CompositeGraph cg(graph_, dg);
  SPDLOG_DEBUG("Generate composite_graph");
  TransitionGraph tg(tc, config.gps_error);
  SPDLOG_DEBUG("Update cost in transition graph");
  // The network will be used internally to update transition graph
  update_tg(&tg, cg, traj, config);
  SPDLOG_DEBUG("Optimal path inference");
  std::vector<TGOpath> tg_opaths = tg.backtrack();
  SPDLOG_DEBUG("Optimal path size {}", tg_opaths[0].size());
  std::vector<MatchedCandidatePath> matched_candidate_paths = build_matched_candidate_paths(tg_opaths);
  std::vector<O_Path> opaths = build_opaths(tg_opaths);
  std::vector<std::vector<int>> indices;
  std::vector<C_Path> cpaths = build_cpaths(tg_opaths, &indices, config.reverse_tolerance);
  SPDLOG_DEBUG("Opath is {}", opaths[0]);
  SPDLOG_DEBUG("Indices is {}", indices[0]);
  SPDLOG_DEBUG("Complete path is {}", cpaths[0]);
  std::vector<MatchResult> match_results;
  int loop_counter = 0;
  for (C_path cpath:cpaths){
    LineString mgeom = network_.complete_path_to_geometry(
      traj.geom, cpath);
    MatchResult match_result{
      traj.id, 
      matched_candidate_paths[loop_counter], 
      opath[loop_counter],
      cpath,
      indices[loop_counter],
      mgeom
    };
    match_results.push_back()
    loop_counter++;
  }

  return match_results;
}

std::string STMATCH::match_gps_file(
  const FMM::CONFIG::GPSConfig &gps_config,
  const FMM::CONFIG::ResultConfig &result_config,
  const STMATCHConfig &stmatch_config,
  bool use_omp
  ){
  std::ostringstream oss;
  std::string status;
  bool validate = true;
  if (!gps_config.validate()) {
    oss<<"gps_config invalid\n";
    validate = false;
  }
  if (!result_config.validate()) {
    oss<<"result_config invalid\n";
    validate = false;
  }
  if (!stmatch_config.validate()) {
    oss<<"stmatch_config invalid\n";
    validate = false;
  }
  if (!validate) {
    oss<<"match_gps_file canceled\n";
    return oss.str();
  }
  // Start map matching
  int progress = 0;
  int points_matched = 0;
  int total_points = 0;
  int step_size = 1000;
  auto begin_time = UTIL::get_current_time();
  FMM::IO::GPSReader reader(gps_config);
  FMM::IO::CSVMatchResultWriter writer(result_config.file,
                                       result_config.output_config);
  if (use_omp) {
    int buffer_trajectories_size = 100000;
    while (reader.has_next_trajectory()) {
      std::vector<Trajectory> trajectories =
        reader.read_next_N_trajectories(buffer_trajectories_size);
      int trajectories_fetched = trajectories.size();
      #pragma omp parallel for
      for (int i = 0; i < trajectories_fetched; ++i) {
        Trajectory &trajectory = trajectories[i];
        int points_in_tr = trajectory.geom.get_num_points();
        MM::MatchResult result = match_traj(
          trajectory, stmatch_config);
        writer.write_result(trajectory,result);
        #pragma omp critical
        if (!result.cpath.empty()) {
          points_matched += points_in_tr;
        }
        total_points += points_in_tr;
        ++progress;
        if (progress % step_size == 0) {
          std::stringstream buf;
          buf << "Progress " << progress << '\n';
          std::cout << buf.rdbuf();
        }
      }
    }
  } else {
    while (reader.has_next_trajectory()) {
      if (progress % step_size == 0) {
        SPDLOG_INFO("Progress {}", progress);
      }
      Trajectory trajectory = reader.read_next_trajectory();
      int points_in_tr = trajectory.geom.get_num_points();
      MM::MatchResult result = match_traj(
        trajectory, stmatch_config);
      writer.write_result(trajectory,result);
      if (!result.cpath.empty()) {
        points_matched += points_in_tr;
      }
      total_points += points_in_tr;
      ++progress;
    }
  }
  auto end_time = UTIL::get_current_time();
  double duration = UTIL::get_duration(begin_time,end_time);
  oss<<"Status: success\n";
  oss<<"Time takes " << duration << " seconds\n";
  oss<<"Total points " << total_points << " matched "<< points_matched <<"\n";
  oss<<"Map match speed " << points_matched / duration << " points/s \n";
  return oss.str();
};

void STMATCH::update_tg(TransitionGraph *tg,
                        const CompositeGraph &cg,
                        const Trajectory &traj,
                        const STMATCHConfig &config) {
  SPDLOG_DEBUG("Update transition graph");
  std::vector<TGLayer> &layers = tg->get_layers();
  std::vector<double> eu_dists = ALGORITHM::cal_eu_dist(traj.geom);
  int N = layers.size();
  for (int i = 0; i < N - 1; ++i) {
    // Routing from current_layer to next_layer
    double delta = 0;
    if (traj.timestamps.size() != N) {
      delta = eu_dists[i] * config.factor * 4;
    } else {
      double duration = traj.timestamps[i + 1] - traj.timestamps[i];
      delta = config.factor * config.vmax * duration;
    }
    update_layer(i, &(layers[i]), &(layers[i + 1]),
                 cg, eu_dists[i], delta);
  }
  SPDLOG_DEBUG("Update transition graph done");
}

void STMATCH::update_layer(int level, TGLayer *la_ptr, TGLayer *lb_ptr,
                           const CompositeGraph &cg,
                           double eu_dist,
                           double delta) {
  SPDLOG_DEBUG("Update layer {} starts", level);
  TGLayer &lb = *lb_ptr;
  for (auto iter_a = la_ptr->begin(); iter_a != la_ptr->end(); ++iter_a) {
    NodeIndex source = iter_a->c->index;
    // SPDLOG_TRACE("  Calculate distance from source {}", source);
    // single source upper bound routing
    std::vector<NodeIndex> targets(lb.size());
    std::transform(lb.begin(), lb.end(), targets.begin(),
                   [](TGNode &a) {
      return a.c->index;
    });
    std::vector<double> distances = shortest_path_upperbound(
      level, cg, source, targets, delta);
    for (auto iter_b = lb_ptr->begin(); iter_b != lb_ptr->end(); ++iter_b) {
      int i = std::distance(lb_ptr->begin(),iter_b);
      double tp = TransitionGraph::calc_tp(distances[i], eu_dist);
      double temp = iter_a->cumu_prob + log(tp) + log(iter_b->ep);
      SPDLOG_TRACE("L {} f {} t {} sp {} dist {} tp {} ep {} fcp {} tcp {}",
        level, iter_a->c->edge->id,iter_b->c->edge->id,
        distances[i], eu_dist, tp, iter_b->ep, iter_a->cumu_prob,
        temp);
      if (temp >= iter_b->cumu_prob) {
        iter_b->cumu_prob = temp;
        iter_b->prev = &(*iter_a);
        iter_b->sp_dist = distances[i];
        iter_b->tp = tp;
      }
    }
  }
  SPDLOG_DEBUG("Update layer done");
}

std::vector<double> STMATCH::shortest_path_upperbound(
  int level, const CompositeGraph &cg, NodeIndex source,
  const std::vector<NodeIndex> &targets, double delta) {
  // SPDLOG_TRACE("Upperbound shortest path source {}", source);
  // SPDLOG_TRACE("Upperbound shortest path targets {}", targets);
  std::unordered_set<NodeIndex> unreached_targets;
  for (auto &node:targets) {
    unreached_targets.insert(node);
  }
  DistanceMap dmap;
  PredecessorMap pmap;
  Heap Q;
  Q.push(source, 0);
  pmap.insert({source, source});
  dmap.insert({source, 0});
  double temp_dist = 0;
  // Dijkstra search
  while (!Q.empty() && !unreached_targets.empty()) {
    HeapNode node = Q.top();
    Q.pop();
    // SPDLOG_TRACE("  Node u {} dist {}", node.index, node.value);
    NodeIndex u = node.index;
    auto iter = unreached_targets.find(u);
    if (iter != unreached_targets.end()) {
      // Remove u
      // SPDLOG_TRACE("  Remove target {}", u);
      unreached_targets.erase(iter);
    }
    if (node.value > delta) break;
    std::vector<CompEdgeProperty> out_edges = cg.out_edges(u);
    for (auto node_iter = out_edges.begin(); node_iter != out_edges.end();
         ++node_iter) {
      NodeIndex v = node_iter->v;
      temp_dist = node.value + node_iter->cost;
      // SPDLOG_TRACE("  Examine node v {} temp dist {}", v, temp_dist);
      auto v_iter = dmap.find(v);
      if (v_iter != dmap.end()) {
        // dmap contains node v
        if (v_iter->second - temp_dist > 1e-6) {
          // a smaller distance is found for v
          // SPDLOG_TRACE("    Update key {} {} in pdmap prev dist {}",
          //              v, temp_dist, v_iter->second);
          pmap[v] = u;
          dmap[v] = temp_dist;
          Q.decrease_key(v, temp_dist);
        }
      } else {
        // dmap does not contain v
        if (temp_dist <= delta) {
          // SPDLOG_TRACE("    Insert key {} {} into pmap and dmap",
          //              v, temp_dist);
          Q.push(v, temp_dist);
          pmap.insert({v, u});
          dmap.insert({v, temp_dist});
        }
      }
    }
  }
  // Update distances
  // SPDLOG_TRACE("  Update distances");
  std::vector<double> distances;
  for (int i = 0; i < targets.size(); ++i) {
    if (dmap.find(targets[i]) != dmap.end()) {
      distances.push_back(dmap[targets[i]]);
    } else {
      distances.push_back(std::numeric_limits<double>::max());
    }
  }
  // SPDLOG_TRACE("  Distance value {}", distances);
  return distances;
}

std::vector<MatchedCandidatePath> STMATCH::build_matched_candidate_paths(
  const std::vector<TGOpath> &tg_opaths
  ){
  std::vector<MatchedCandidatePath> matched_candidate_paths;
  for (TGOpath tg_opath:tg_opaths){
    MatchedCandidatePath matched_candidate_path(tg_opath.size());
    std::transform(tg_opath.begin(), tg_opath.end(),
                  matched_candidate_path.begin(),
                  [](const TGNode *a) {
      return MatchedCandidate{
        *(a->c), a->ep, a->tp, a->sp_dist
      };
    });
    matched_candidate_paths.push_back(matched_candidate_path);
  }
  return matched_candidate_paths;
}

std::vector<O_Path> STMATCH::build_opaths(
  const std::vector<TGOpath> &tg_opaths
  ){
  std::vector<O_Path> opaths;
  for (TGOpath tg_opath:tg_opaths){
    O_Path opath(tg_opath.size());
    std::transform(tg_opath.begin(), tg_opath.end(),
                 opath.begin(),
                 [](const TGNode *a) {
      return a->c->edge->id;
    });
    opaths.push_back(opath);
  }
  return opaths;
}

std::vector<C_Path> STMATCH::build_cpaths(const std::vector<TGOpath> &opaths, std::vector<std::vector<int>> *all_indices,
  double reverse_tolerance) {
  SPDLOG_DEBUG("Build cpath from optimal candidate path");
  std::vector<C_Path> cpaths;
  for (TGOpath opath:opaths){
    C_path cpath;
    std::vector<int> indices;
    if (!opath.empty()){
      const std::vector<Edge> &edges = network_.get_edges();
      int N = opath.size();
      cpath.push_back(opath[0]->c->edge->id);
      int current_idx = 0;
      // SPDLOG_TRACE("Insert index {}", current_idx);
      indices.push_back(current_idx);
      for (int i = 0; i < N - 1; ++i) {
        const Candidate *a = opath[i]->c;
        const Candidate *b = opath[i + 1]->c;
        // SPDLOG_TRACE("Check a {} b {}", a->edge->id, b->edge->id);
        if ((a->edge->id != b->edge->id) ||
            (a->offset-b->offset > a->edge->length * reverse_tolerance)) {
          auto segs = graph_.shortest_path_dijkstra(a->edge->target,
                                                    b->edge->source);
          // No transition found
          if (segs.empty() && a->edge->target != b->edge->source) {
            SPDLOG_TRACE("Candidate {} has disconnected edge {} to {}",
              i, a->edge->id, b->edge->id);
            indices.clear();
            return {};
          }
          for (int e:segs) {
            cpath.push_back(edges[e].id);
            ++current_idx;
          }
          cpath.push_back(b->edge->id);
          ++current_idx;
          indices.push_back(current_idx);
        } else {
          indices.push_back(current_idx);
        }
      }
    }
    all_indices->push_back(indices);
    cpaths.push_back(cpath);
  }
  SPDLOG_DEBUG("Build cpath from optimal candidate path done");
  return cpaths;
}
