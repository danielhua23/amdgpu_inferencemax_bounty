// ============================================
// InferenceMAX Benchmark Script - GPT-OSS FP4 Multi-Concurrency Mode (C++ Version)
// ============================================
// Compile with:
//   g++ -std=c++17 -o gptoss_benchmark gptoss_benchmark.cpp -lcurl -pthread -O2
//
// Usage:
//   ./gptoss_benchmark acc                           # Run accuracy test only
//   ./gptoss_benchmark perf                          # Run accuracy + performance tests
//   ./gptoss_benchmark submit <team>                 # Run all tests + submit to leaderboard
//   ./gptoss_benchmark acc -isl 1024 -osl 1024       # Test CONC=4,8
//   ./gptoss_benchmark submit <team> -isl 1024 -osl 8192  # Test all CONC + submit

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <regex>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <libgen.h>
#include <limits.h>
#include <cmath>

using namespace std;

// ============================================
// Configuration Structure
// ============================================
struct Config {
    string mode;
    string team_name;
    string isl_arg;
    string osl_arg;
    
    // Environment variables
    string model;
    int port = 8888;
    int tp = 8;
    int conc = 8;  // GPT-OSS default is 8
    int isl = 1024;
    int osl = 1024;
    int max_model_len = 16384;  // GPT-OSS specific
    double random_range_ratio = 1.0;
    string result_filename = "result";
    int num_prompts = 0;
    
    string lb_url_override;
    bool multi_conc_mode = false;
    string script_path;
    string script_dir;
};

// ============================================
// Baseline Data Structure (GPT-OSS)
// ============================================
struct Baseline {
    double median_e2e;
    double median_intvty;
    double tput_per_gpu;
};

map<string, Baseline> BASELINES = {
    // ISL-OSL=1024-1024
    {"1024_1024_4", {2162, 434.57, 413.779}},
    {"1024_1024_8", {2646, 364.58, 690.565}},
    
    // ISL-OSL=1024-8192
    {"1024_8192_4", {16585, 433.22, 241.18}},
    {"1024_8192_8", {19939, 374.06, 407.268}},
    {"1024_8192_16", {22496, 332.04, 719.886}},
    
    // ISL-OSL=8192-1024
    {"8192_1024_4", {2275, 408.40, 1752.338}},
    {"8192_1024_8", {2792, 342.70, 2892.223}},
};

// ============================================
// Utility Functions
// ============================================

string get_executable_path() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        return string(result);
    }
    return "";
}

string get_executable_dir() {
    string exe_path = get_executable_path();
    if (exe_path.empty()) {
        return "";
    }
    // Get directory path
    char* path_copy = strdup(exe_path.c_str());
    char* dir = dirname(path_copy);
    string result(dir);
    free(path_copy);
    return result;
}

string get_env_var(const string& name, const string& default_value = "") {
    const char* val = getenv(name.c_str());
    return val ? string(val) : default_value;
}

void set_env_var(const string& name, const string& value) {
    setenv(name.c_str(), value.c_str(), 1);
}

string get_timestamp() {
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

string get_current_time_str() {
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&time_t), "%c");
    return ss.str();
}

int execute_command(const string& cmd, string* output = nullptr, bool show_output = true) {
    if (show_output) {
        cout << "Executing: " << cmd << endl;
    }
    
    if (output) {
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return -1;
        
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            *output += buffer;
            if (show_output) {
                cout << buffer;
            }
        }
        
        int status = pclose(pipe);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    } else {
        return system(cmd.c_str());
    }
}

bool file_exists(const string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool create_directory(const string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

string extract_regex_match(const string& text, const regex& pattern, int group = 1) {
    smatch match;
    if (regex_search(text, match, pattern) && match.size() > static_cast<size_t>(group)) {
        return match[group].str();
    }
    return "";
}

string get_leaderboard_url(const string& isl, const string& osl) {
    string key = isl + "_" + osl;
    if (key == "1024_1024") {
        return "https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space";
    } else if (key == "1024_8192") {
        return "https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space";
    } else if (key == "8192_1024") {
        return "https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space";
    } else {
        return "ERROR: wrong isl and osl config, pls check";
    }
}

// ============================================
// Simple JSON Parser
// ============================================
class SimpleJSON {
private:
    map<string, string> data;
    
public:
    void parse_simple_json(const string& json) {
        regex pattern("\"([^\"]+)\"\\s*:\\s*([0-9.]+|\"[^\"]*\"|null)");
        smatch match;
        string::const_iterator searchStart(json.cbegin());
        
        while (regex_search(searchStart, json.cend(), match, pattern)) {
            string key = match[1].str();
            string value = match[2].str();
            data[key] = value;
            searchStart = match.suffix().first;
        }
    }
    
    double get_double(const string& key, double default_val = 0.0) {
        if (data.find(key) != data.end()) {
            string val = data[key];
            if (val == "null") return default_val;
            try {
                return stod(val);
            } catch (...) {
                return default_val;
            }
        }
        return default_val;
    }
    
    string get_string(const string& key, const string& default_val = "") {
        if (data.find(key) != data.end()) {
            string val = data[key];
            if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
                return val.substr(1, val.size() - 2);
            }
            return val;
        }
        return default_val;
    }
};

// ============================================
// Run Benchmark Serving Function
// ============================================
int run_benchmark_serving(const Config& cfg) {
    cout << "INFO: Starting performance benchmark..." << endl;
    
    string bench_dir = "/tmp/bmk-" + to_string(time(nullptr));
    string clone_cmd = "git clone https://github.com/kimbochen/bench_serving.git " + bench_dir;
    if (execute_command(clone_cmd) != 0) {
        cerr << "ERROR: Failed to clone benchmark_serving repo" << endl;
        return 1;
    }
    
    stringstream cmd;
    cmd << "python3 " << bench_dir << "/benchmark_serving.py"
        << " --model \"" << cfg.model << "\""
        << " --backend vllm"
        << " --base-url http://0.0.0.0:" << cfg.port
        << " --dataset-name random"
        << " --random-input-len " << cfg.isl
        << " --random-output-len " << cfg.osl
        << " --random-range-ratio " << cfg.random_range_ratio
        << " --num-prompts " << cfg.num_prompts
        << " --max-concurrency " << cfg.conc
        << " --request-rate inf"
        << " --ignore-eos"
        << " --save-result"
        << " --percentile-metrics 'ttft,tpot,itl,e2el'"
        << " --result-dir " << cfg.script_dir << "/"
        << " --result-filename " << cfg.result_filename << ".json";
    
    int ret = execute_command(cmd.str());
    execute_command("rm -rf " + bench_dir, nullptr, false);
    
    return ret;
}

// ============================================
// Server Health Check
// ============================================
bool check_server_health(const Config& cfg, int max_retries = 5, int delay_seconds = 3) {
    cout << "INFO: Checking server health at http://0.0.0.0:" << cfg.port << endl;
    
    for (int attempt = 1; attempt <= max_retries; attempt++) {
        stringstream cmd;
        cmd << "curl -s -o /dev/null -w '%{http_code}' "
            << "http://0.0.0.0:" << cfg.port << "/health 2>/dev/null";
        
        string output;
        int ret = execute_command(cmd.str(), &output, false);
        
        // Remove whitespace
        output.erase(remove_if(output.begin(), output.end(), ::isspace), output.end());
        
        if (output == "200" || output.find("200") != string::npos) {
            cout << "SUCCESS: Server is healthy and responding" << endl;
            return true;
        }
        
        cout << "WARNING: Server health check failed (attempt " << attempt << "/" << max_retries 
             << "), response: " << output << endl;
        
        if (attempt < max_retries) {
            cout << "INFO: Waiting " << delay_seconds << " seconds before retry..." << endl;
            sleep(delay_seconds);
        }
    }
    
    cerr << "ERROR: Server health check failed after " << max_retries << " attempts" << endl;
    cerr << "ERROR: Please ensure vLLM server is running on port " << cfg.port << endl;
    return false;
}

// ============================================
// Run Accuracy Test
// ============================================
struct AccuracyMetrics {
    // GPQA metric reported by `python -m gpt_oss.evals --eval gpqa ...`
    // Parsed from stdout as: "'gpqa_metric': <float>"
    double gpqa_metric = 0.0;
};

int run_accuracy_test_gpqa(const Config& cfg, AccuracyMetrics& metrics) {
    cout << "INFO: Starting accuracy test (GPQA via gpt_oss.evals)" << endl;
    
    // Check server health first
    if (!check_server_health(cfg)) {
        cerr << "ERROR: Server is not responding. Cannot proceed with accuracy test." << endl;
        return 1;
    }
    
    string script_path = cfg.script_dir + "/accuracy_test_gpqa.py";
    if (!file_exists(script_path)) {
        cerr << "ERROR: GPQA accuracy test script not found: " << script_path << endl;
        cerr << "ERROR: Please ensure accuracy_test_gpqa.py exists in the same directory" << endl;
        return 1;
    }
    
    cout << "INFO: Running GPQA evaluation (OpenAI-compatible API)" << endl;
    
    stringstream cmd;
    cmd << "python3 " << script_path
        << " --base-url http://0.0.0.0:" << cfg.port
        << " --model " << cfg.model
        << " 2>&1";
    
    string output;
    int ret = execute_command(cmd.str(), &output);
    
    if (ret != 0) {
        cerr << "\nERROR: Alternative accuracy test failed with exit code " << ret << endl;
        return ret;
    }
    
    // Parse output
    regex metric_pattern(R"(gpqa_metric[^0-9]*([0-9]+(?:\.[0-9]+)?))");
    string metric_str = extract_regex_match(output, metric_pattern);
    metrics.gpqa_metric = metric_str.empty() ? 0.0 : stod(metric_str);
    
    cout << "INFO: Accuracy metrics:" << endl;
    cout << "  GPQA metric: " << metrics.gpqa_metric << endl;
    
    return 0;
}

int run_accuracy_test(const Config& cfg, AccuracyMetrics& metrics) {
    return run_accuracy_test_gpqa(cfg, metrics);
}

// ============================================
// Validate Accuracy Metrics (GPT-OSS Baselines)
// ============================================
int validate_accuracy(const AccuracyMetrics& metrics) {
    // Default baseline matches vLLM's GPT-OSS GPQA test example.
    // Override via environment variables if you need a different threshold.
    const double BASELINE_GPQA_METRIC = stod(get_env_var("GPQA_BASELINE_METRIC", "0.58"));
    const double GPQA_TOL = stod(get_env_var("GPQA_TOL", "0.05"));  // absolute tolerance
    const double MIN_ACCEPTED = BASELINE_GPQA_METRIC - GPQA_TOL;
    
    cout << "\nINFO: Validating GPQA metric against baseline..." << endl;
    cout << "  Baseline gpqa_metric=" << BASELINE_GPQA_METRIC << endl;
    cout << "  Tolerance (absolute): " << GPQA_TOL << endl;
    cout << "  Minimum accepted: " << MIN_ACCEPTED << endl;
    
    if (metrics.gpqa_metric < MIN_ACCEPTED) {
        cout << "\nERROR: Accuracy validation FAILED!" << endl;
        cout << "ERROR: gpqa_metric too low: " << metrics.gpqa_metric
             << " < " << MIN_ACCEPTED << endl;
        cout << "\nERROR: Performance benchmark will NOT be run due to accuracy validation failure." << endl;
        cout << "ERROR: Please investigate and improve model accuracy before proceeding." << endl;
        return 1;
    }
    
    cout << "\nSUCCESS: Accuracy validation PASSED!" << endl;
    cout << "✓ gpqa_metric: " << metrics.gpqa_metric << " ≥ " << MIN_ACCEPTED << endl;
    cout << endl;
    return 0;
}

// ============================================
// Process Result JSON and Add Metrics
// ============================================
int process_result_json(const Config& cfg, const AccuracyMetrics& acc_metrics) {
    string result_file = cfg.script_dir + "/" + cfg.result_filename + ".json";
    
    if (!file_exists(result_file)) {
        cerr << "WARNING: Result file " << result_file << " not found, cannot add metrics" << endl;
        return 1;
    }
    
    cout << "INFO: Cleaning up and adding metrics to " << result_file << endl;
    
    string python_script = R"(
import json
import sys

# Baseline Data (NV-1126) - GPT-OSS
BASELINES = {
    # ISL-OSL=1024-1024
    ('1024', '1024', '4'): {'median_e2e': 2162, 'median_intvty': 434.57,'tput_per_gpu': 413.779},
    ('1024', '1024', '8'): {'median_e2e': 2646, 'median_intvty': 364.58,'tput_per_gpu': 690.565},
    
    # ISL-OSL=1024-8192
    ('1024', '8192', '4'): {'median_e2e': 16585, 'median_intvty': 433.22,'tput_per_gpu': 241.18},
    ('1024', '8192', '8'): {'median_e2e': 19939, 'median_intvty': 374.06,'tput_per_gpu': 407.268},
    ('1024', '8192', '16'): {'median_e2e': 22496, 'median_intvty': 332.04,'tput_per_gpu': 719.886},
    
    # ISL-OSL=8192-1024
    ('8192', '1024', '4'): {'median_e2e': 2275, 'median_intvty': 408.40,'tput_per_gpu': 1752.338},
    ('8192', '1024', '8'): {'median_e2e': 2792, 'median_intvty': 342.70,'tput_per_gpu': 2892.223},
}

result_file = sys.argv[1]
isl = sys.argv[2]
osl = sys.argv[3]
conc = sys.argv[4]
model = sys.argv[5]
port = sys.argv[6]
random_range_ratio = sys.argv[7]
num_prompts = sys.argv[8]
import os
gpqa_metric = float(sys.argv[9])
baseline_gpqa_metric = float(os.environ.get('GPQA_BASELINE_METRIC', '0.58'))
gpqa_tol = float(os.environ.get('GPQA_TOL', '0.05'))

try:
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    summary_data = {}
    
    keep_fields = [
        'successful_requests', 'benchmark_duration', 'total_input_tokens',
        'total_generated_tokens', 'request_throughput', 'output_throughput',
        'total_token_throughput', 'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
        'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms',
        'median_itl_ms', 'p99_itl_ms', 'mean_e2el_ms', 'median_e2el_ms', 'p99_e2el_ms'
    ]
    
    for field in keep_fields:
        if field in data:
            summary_data[field] = data[field]
    
    summary_data['benchmark_args'] = {
        'model': model, 'backend': 'vllm', 'base_url': f'http://0.0.0.0:{port}',
        'dataset_name': 'random', 'random_input_len': int(isl),
        'random_output_len': int(osl), 'random_range_ratio': float(random_range_ratio),
        'num_prompts': int(num_prompts), 'max_concurrency': int(conc),
        'request_rate': 'inf'
    }
    
    mi355x_tput_per_gpu = 0.0
    mi355x_median_e2e = 0.0
    
    if 'total_token_throughput' in data:
        mi355x_tput_per_gpu = data['total_token_throughput'] / 8.0
        summary_data['tput_per_gpu'] = mi355x_tput_per_gpu
    
    if 'median_e2el_ms' in data:
        mi355x_median_e2e = data['median_e2el_ms']
    
    if 'median_tpot_ms' in data and data['median_tpot_ms'] > 0:
        summary_data['interactivity'] = 1000.0 / data['median_tpot_ms']
    else:
        summary_data['interactivity'] = 0.0
    
    baseline_key = (isl, osl, conc)
    if baseline_key in BASELINES:
        baseline_data = BASELINES[baseline_key]
        summary_data['baseline_nv1126'] = {
            'baseline_median_e2e_1126': baseline_data['median_e2e'],
            'baseline_tput_pergpu_1126': baseline_data['tput_per_gpu'],
            'baseline_median_intvty_1126': baseline_data['median_intvty']
        }
        
        if baseline_data['tput_per_gpu'] > 0:
            summary_data['tput_per_gpu_ratio_vs_baseline_1126'] = mi355x_tput_per_gpu / baseline_data['tput_per_gpu']
        else:
            summary_data['tput_per_gpu_ratio_vs_baseline_1126'] = 0.0
        
        if baseline_data['median_e2e'] > 0:
            summary_data['median_e2e_ratio_vs_baseline_1126'] = mi355x_median_e2e / baseline_data['median_e2e']
        else:
            summary_data['median_e2e_ratio_vs_baseline_1126'] = 0.0
        
        if baseline_data['median_intvty'] > 0:
            summary_data['interactivity_ratio_vs_baseline_1126'] = summary_data['interactivity'] / baseline_data['median_intvty']
        else:
            summary_data['interactivity_ratio_vs_baseline_1126'] = 0.0
        
        print(f'INFO: Baseline found for ISL={isl}, OSL={osl}, CONC={conc}')
        print(f'INFO: Throughput ratio (MI355X/baseline, higher is better!): {summary_data["tput_per_gpu_ratio_vs_baseline_1126"]:.4f}')
        print(f'INFO: E2E latency ratio (MI355X/baseline, lower is better!): {summary_data["median_e2e_ratio_vs_baseline_1126"]:.4f}')
        print(f'INFO: Interactivity: {summary_data["interactivity"]:.2f} tokens/s/user')
        print(f'INFO: Interactivity ratio (MI355X/baseline, higher is better!): {summary_data["interactivity_ratio_vs_baseline_1126"]:.4f}')
    else:
        print(f'WARNING: No baseline found for ISL={isl}, OSL={osl}, CONC={conc}')
        summary_data['baseline_nv1126'] = None
        summary_data['tput_per_gpu_ratio_vs_baseline_1126'] = None
        summary_data['median_e2e_ratio_vs_baseline_1126'] = None
        summary_data['interactivity_ratio_vs_baseline_1126'] = None
    
    summary_data['accuracy'] = {
        'gpqa_metric': gpqa_metric,
    }
    
    summary_data['accuracy_validation'] = {
        'status': 'PASSED',
        'baselines': {
            'gpqa_metric': baseline_gpqa_metric,
        },
        'tolerance': gpqa_tol
    }
    
    with open(result_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print('INFO: Result file cleaned and all metrics added')
except Exception as e:
    print(f'ERROR: Failed to process result file: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
)";
    
    string script_path = "/tmp/process_json_" + to_string(time(nullptr)) + ".py";
    ofstream script_file(script_path);
    script_file << python_script;
    script_file.close();
    
    stringstream cmd;
    cmd << "python3 " << script_path
        << " " << result_file
        << " " << cfg.isl
        << " " << cfg.osl
        << " " << cfg.conc
        << " \"" << cfg.model << "\""
        << " " << cfg.port
        << " " << cfg.random_range_ratio
        << " " << cfg.num_prompts
        << " " << acc_metrics.gpqa_metric;
    
    int ret = execute_command(cmd.str());
    remove(script_path.c_str());
    
    return ret;
}

// ============================================
// Submit to Leaderboard
// ============================================
int submit_to_leaderboard(const Config& cfg, const string& lb_url) {
    cout << "\n============================================" << endl;
    cout << "Submitting results to leaderboard" << endl;
    cout << "============================================" << endl;
    cout << "Team: " << cfg.team_name << endl;
    cout << "Leaderboard URL: " << lb_url << endl;
    cout << "Configuration: ISL=" << cfg.isl << ", OSL=" << cfg.osl << ", CONC=" << cfg.conc << endl;
    
    string result_file = cfg.script_dir + "/" + cfg.result_filename + ".json";
    
    if (!file_exists(result_file)) {
        cerr << "ERROR: Result file not found: " << result_file << endl;
        return 1;
    }
    
    string python_parse_script = R"(
import json
import sys

try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    mi355x_e2e = data.get('median_e2el_ms', 0.0)
    mi355x_throughput = data.get('tput_per_gpu', 0.0)
    median_tpot_ms = data.get('median_tpot_ms', 0.0)
    
    if median_tpot_ms > 0:
        interactivity = 1000.0 / median_tpot_ms
    else:
        interactivity = 0.0
    
    baseline = data.get('baseline_nv1126', {})
    if baseline:
        baseline_e2e = baseline.get('baseline_median_e2e_1126', 0.0)
        baseline_throughput = baseline.get('baseline_tput_pergpu_1126', 0.0)
        baseline_interactivity = baseline.get('baseline_median_intvty_1126', 0.0)
    else:
        baseline_e2e = 0.0
        baseline_throughput = 0.0
        baseline_interactivity = 0.0
    
    e2e_ratio = data.get('median_e2e_ratio_vs_baseline_1126', 0.0)
    throughput_ratio = data.get('tput_per_gpu_ratio_vs_baseline_1126', 0.0)
    interactivity_ratio = data.get('interactivity_ratio_vs_baseline_1126', 0.0)
    
    if e2e_ratio is None: e2e_ratio = 0.0
    if throughput_ratio is None: throughput_ratio = 0.0
    if interactivity_ratio is None: interactivity_ratio = 0.0
    
    acc = data.get('accuracy', {})
    gpqa_metric = acc.get('gpqa_metric', 0.0)
    
    print(f'{mi355x_e2e},{mi355x_throughput},{baseline_e2e},{baseline_throughput},{e2e_ratio},{throughput_ratio},{interactivity},{baseline_interactivity},{interactivity_ratio},{gpqa_metric}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
)";
    
    string parse_script_path = "/tmp/parse_metrics_" + to_string(time(nullptr)) + ".py";
    ofstream parse_file(parse_script_path);
    parse_file << python_parse_script;
    parse_file.close();
    
    string metrics_output;
    stringstream parse_cmd;
    parse_cmd << "python3 " << parse_script_path << " " << result_file;
    
    if (execute_command(parse_cmd.str(), &metrics_output, false) != 0) {
        cerr << "ERROR: Failed to parse metrics from result file" << endl;
        remove(parse_script_path.c_str());
        return 1;
    }
    
    remove(parse_script_path.c_str());
    
    stringstream ss(metrics_output);
    vector<string> values;
    string token;
    while (getline(ss, token, ',')) {
        values.push_back(token);
    }
    
    if (values.size() < 10) {
        cerr << "ERROR: Invalid metrics output" << endl;
        return 1;
    }
    
    double mi355x_e2e = stod(values[0]);
    double mi355x_throughput = stod(values[1]);
    double baseline_e2e = stod(values[2]);
    double baseline_throughput = stod(values[3]);
    double e2e_ratio = stod(values[4]);
    double throughput_ratio = stod(values[5]);
    double interactivity = stod(values[6]);
    double baseline_interactivity = stod(values[7]);
    double interactivity_ratio = stod(values[8]);
    double gpqa_metric = stod(values[9]);
    
    cout << "\nConfiguration:" << endl;
    cout << "  CONC: " << cfg.conc << endl;
    cout << "\nMI355X Performance:" << endl;
    cout << "  E2E (median): " << mi355x_e2e << "ms" << endl;
    cout << "  Throughput per GPU: " << mi355x_throughput << " tokens/s" << endl;
    cout << "  Interactivity: " << interactivity << " tokens/s/user" << endl;
    cout << "\nBaseline (NV-1126):" << endl;
    cout << "  E2E (median): " << baseline_e2e << "ms" << endl;
    cout << "  Throughput per GPU: " << baseline_throughput << " tokens/s" << endl;
    cout << "  Interactivity: " << baseline_interactivity << " tokens/s/user" << endl;
    cout << "\nPerformance Ratios (MI355X / baseline):" << endl;
    cout << "  E2E Ratio: " << e2e_ratio << endl;
    cout << "  Throughput Ratio: " << throughput_ratio << endl;
    cout << "  Interactivity Ratio: " << interactivity_ratio << endl;
    cout << "\nAccuracy metrics:" << endl;
    cout << "  GPQA metric: " << gpqa_metric << endl;
    
    cout << "\nSubmitting to leaderboard..." << endl;
    
    stringstream curl_cmd;
    curl_cmd << "curl -X POST " << lb_url << "/gradio_api/call/submit_results -s "
             << "-H \"Content-Type: application/json\" "
             << "-d '{\"data\": ["
             << "\"" << cfg.team_name << "\", "
             << cfg.conc << ", "
             << mi355x_e2e << ", "
             << mi355x_throughput << ", "
             << baseline_e2e << ", "
             << baseline_throughput << ", "
             << e2e_ratio << ", "
             << throughput_ratio << ", "
             << interactivity << ", "
             << baseline_interactivity << ", "
             << interactivity_ratio << ", "
             << gpqa_metric
             << "]}'";
    
    string submit_response;
    execute_command(curl_cmd.str(), &submit_response, false);
    
    sleep(2);
    
    cout << "\n============================================" << endl;
    cout << "SUCCESS: Results submitted to leaderboard! 🎉" << endl;
    cout << "Check it out @ " << lb_url << endl;
    cout << "============================================" << endl;
    
    return 0;
}

// ============================================
// Run Single Configuration Test
// ============================================
int run_single_test(Config cfg, const AccuracyMetrics& acc_metrics) {
    cout << "============================================" << endl;
    cout << "Mode: " << cfg.mode << endl;
    if (cfg.mode == "submit") {
        cout << "Team: " << cfg.team_name << endl;
        
        string lb_url;
        if (!cfg.lb_url_override.empty()) {
            lb_url = cfg.lb_url_override;
        } else {
            lb_url = get_leaderboard_url(to_string(cfg.isl), to_string(cfg.osl));
        }
        cout << "Leaderboard: " << lb_url << endl;
    }
    cout << "============================================" << endl;
    
    if (cfg.mode == "acc") {
        cout << "\n============================================" << endl;
        cout << "Mode: acc - Accuracy test completed" << endl;
        cout << "============================================" << endl;
        cout << "Accuracy metrics:" << endl;
        cout << "  GPQA metric: " << acc_metrics.gpqa_metric << endl;
        cout << "\nSUCCESS: Accuracy test completed successfully!" << endl;
        cout << "Skipping performance benchmark (acc mode)" << endl;
        return 0;
    }
    
    if (run_benchmark_serving(cfg) != 0) {
        cerr << "ERROR: Performance benchmark failed" << endl;
        return 1;
    }
    
    if (process_result_json(cfg, acc_metrics) != 0) {
        cerr << "ERROR: Failed to process result JSON" << endl;
        return 1;
    }
    
    if (cfg.mode == "submit") {
        string lb_url;
        if (!cfg.lb_url_override.empty()) {
            lb_url = cfg.lb_url_override;
        } else {
            lb_url = get_leaderboard_url(to_string(cfg.isl), to_string(cfg.osl));
        }
        
        if (submit_to_leaderboard(cfg, lb_url) != 0) {
            cerr << "ERROR: Failed to submit to leaderboard" << endl;
            return 1;
        }
    }
    
    cout << "\nSUCCESS: All tests completed successfully!" << endl;
    return 0;
}

// ============================================
// Run Multi-Concurrency Mode (GPT-OSS specific CONC values)
// ============================================
int run_multi_conc_mode(Config cfg) {
    cout << "============================================" << endl;
    cout << "Multi-Concurrency Testing Mode" << endl;
    cout << "============================================" << endl;
    cout << "ISL: " << cfg.isl_arg << endl;
    cout << "OSL: " << cfg.osl_arg << endl;
    cout << "Mode: " << cfg.mode << endl;
    
    // GPT-OSS specific: determine CONC values based on ISL-OSL
    vector<int> conc_values;
    // if (cfg.isl_arg == "1024" && cfg.osl_arg == "8192") {
    //     conc_values = {4, 8, 16};
    //     cout << "CONC values: 4, 8, 16" << endl;
    // } else {
    //     conc_values = {4, 8};
    //     cout << "CONC values: 4, 8" << endl;
    // }
    conc_values = {4, 8, 16, 32, 64, 256}; // FOR 128, ATOM can beat g200 trt
    cout << "CONC values: 4, 8, 16, 32, 64, 256" << endl;
    string lb_url;
    if (cfg.mode == "submit") {
        cout << "Team: " << cfg.team_name << endl;
        lb_url = get_leaderboard_url(cfg.isl_arg, cfg.osl_arg);
        cout << "Leaderboard: " << lb_url << endl;
    }
    cout << "============================================" << endl;
    cout << endl;
    
    string batch_results_dir = "batch_isl" + cfg.isl_arg + "_osl" + cfg.osl_arg + "_" + get_timestamp();
    if (!create_directory(batch_results_dir)) {
        cerr << "ERROR: Failed to create results directory" << endl;
        return 1;
    }
    
    cout << "Results directory: " << batch_results_dir << endl;
    cout << endl;
    
    int passed = 0;
    int failed = 0;
    
    string summary_file = batch_results_dir + "/summary.txt";
    ofstream summary(summary_file);
    summary << "Multi-Concurrency Test Results" << endl;
    summary << "ISL: " << cfg.isl_arg << ", OSL: " << cfg.osl_arg << endl;
    summary << "Mode: " << cfg.mode << endl;
    summary << "Time: " << get_current_time_str() << endl;
    summary << "============================================" << endl;
    summary << endl;
    summary.close();
    
    for (int conc : conc_values) {
        cout << endl;
        cout << "============================================" << endl;
        cout << "Testing CONC=" << conc << endl;
        cout << "============================================" << endl;
        
        // GPT-OSS: NUM_PROMPTS = CONC * 10
        int num_prompts = conc * 10;
        
        string result_filename = "result_isl" + cfg.isl_arg + "_osl" + cfg.osl_arg + "_conc" + to_string(conc);
        
        set_env_var("MODEL", cfg.model);
        set_env_var("PORT", to_string(cfg.port));
        set_env_var("TP", to_string(cfg.tp));
        set_env_var("ISL", cfg.isl_arg);
        set_env_var("OSL", cfg.osl_arg);
        set_env_var("CONC", to_string(conc));
        set_env_var("MAX_MODEL_LEN", to_string(cfg.max_model_len));
        set_env_var("RANDOM_RANGE_RATIO", to_string(cfg.random_range_ratio));
        set_env_var("NUM_PROMPTS", to_string(num_prompts));
        set_env_var("RESULT_FILENAME", batch_results_dir + "/" + result_filename);
        
        if (cfg.mode == "submit") {
            set_env_var("LB_URL_OVERRIDE", lb_url);
        }
        
        auto start_time = chrono::steady_clock::now();
        
        stringstream recursive_cmd;
        recursive_cmd << cfg.script_path;
        if (cfg.mode == "submit") {
            recursive_cmd << " submit \"" << cfg.team_name << "\"";
        } else {
            recursive_cmd << " " << cfg.mode;
        }
        
        int test_status = execute_command(recursive_cmd.str());
        
        auto end_time = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();
        
        ofstream summary_append(summary_file, ios::app);
        if (test_status == 0) {
            passed++;
            string msg = "✓ CONC=" + to_string(conc) + ": PASSED (" + to_string(duration) + "s)";
            cout << msg << endl;
            summary_append << msg << endl;
        } else {
            failed++;
            string msg = "✗ CONC=" + to_string(conc) + ": FAILED (" + to_string(duration) + "s)";
            cout << msg << endl;
            summary_append << msg << endl;
        }
        summary_append.close();
        
        sleep(2);
    }
    
    ofstream summary_final(summary_file, ios::app);
    summary_final << endl;
    summary_final << "============================================" << endl;
    summary_final << "Multi-Concurrency Test Complete!" << endl;
    summary_final << "============================================" << endl;
    summary_final << "Total tests: " << conc_values.size() << endl;
    summary_final << "Passed: " << passed << endl;
    summary_final << "Failed: " << failed << endl;
    summary_final << endl;
    summary_final << "Results saved in: " << batch_results_dir << "/" << endl;
    summary_final << "============================================" << endl;
    summary_final.close();
    
    ifstream summary_read(summary_file);
    string line;
    bool print = false;
    while (getline(summary_read, line)) {
        if (line.empty() && !print) {
            print = true;
            continue;
        }
        if (print) {
            cout << line << endl;
        }
    }
    summary_read.close();
    
    return 0;
}

// ============================================
// Main Function
// ============================================
int main(int argc, char** argv) {
    Config cfg;
    
    cfg.script_path = get_executable_path();
    cfg.script_dir = get_executable_dir();
    
    int i = 1;
    while (i < argc) {
        string arg = argv[i];
        
        if (arg == "acc" || arg == "perf" || arg == "submit") {
            cfg.mode = arg;
            i++;
        } else if (arg == "-isl" || arg == "--isl") {
            if (i + 1 < argc) {
                cfg.isl_arg = argv[i + 1];
                i += 2;
            } else {
                cerr << "ERROR: -isl requires an argument" << endl;
                return 1;
            }
        } else if (arg == "-osl" || arg == "--osl") {
            if (i + 1 < argc) {
                cfg.osl_arg = argv[i + 1];
                i += 2;
            } else {
                cerr << "ERROR: -osl requires an argument" << endl;
                return 1;
            }
        } else {
            if (cfg.mode == "submit" && cfg.team_name.empty()) {
                cfg.team_name = arg;
            }
            i++;
        }
    }
    
    if (cfg.mode.empty()) {
        cfg.mode = "acc";
    }
    
    if (cfg.mode != "acc" && cfg.mode != "perf" && cfg.mode != "submit") {
        cerr << "ERROR: Invalid mode '" << cfg.mode << "'" << endl;
        cerr << "Usage:" << endl;
        cerr << "  " << argv[0] << " acc [-isl <value>] [-osl <value>]" << endl;
        cerr << "  " << argv[0] << " perf [-isl <value>] [-osl <value>]" << endl;
        cerr << "  " << argv[0] << " submit <team> [-isl <value>] [-osl <value>]" << endl;
        return 1;
    }
    
    if (cfg.mode == "submit") {
        if (cfg.team_name.empty()) {
            cfg.team_name = get_env_var("TEAM_NAME_ENV");
            if (cfg.team_name.empty()) {
                cerr << "ERROR: Team name required for submit mode" << endl;
                cerr << "Usage: " << argv[0] << " submit <team_name> [-isl <value>] [-osl <value>]" << endl;
                cerr << "Or set TEAM_NAME_ENV environment variable" << endl;
                return 1;
            }
        }
    }
    
    cfg.multi_conc_mode = !cfg.isl_arg.empty() && !cfg.osl_arg.empty();
    
    if (cfg.multi_conc_mode) {
        cfg.model = get_env_var("MODEL");
        if (cfg.model.empty()) {
            cerr << "ERROR: MODEL environment variable is required" << endl;
            cerr << "Example: export MODEL='openai/gpt-oss-120b'" << endl;
            return 1;
        }
        
        string port_str = get_env_var("PORT");
        if (!port_str.empty()) {
            cfg.port = stoi(port_str);
        } else {
            cout << "WARNING: PORT not set, using default 8888" << endl;
        }
        
        string tp_str = get_env_var("TP");
        if (!tp_str.empty()) {
            cfg.tp = stoi(tp_str);
        } else {
            cout << "WARNING: TP not set, using default 8" << endl;
        }
        
        return run_multi_conc_mode(cfg);
    }
    
    // Single Configuration Mode
    cfg.model = get_env_var("MODEL");
    if (cfg.model.empty()) {
        cerr << "ERROR: MODEL environment variable is not set" << endl;
        cerr << "Example: export MODEL='openai/gpt-oss-120b'" << endl;
        return 1;
    }
    
    string port_str = get_env_var("PORT");
    if (!port_str.empty()) {
        cfg.port = stoi(port_str);
    } else {
        cout << "WARNING: PORT not set, using default 8888" << endl;
    }
    
    string tp_str = get_env_var("TP");
    if (!tp_str.empty()) {
        cfg.tp = stoi(tp_str);
    } else {
        cout << "WARNING: TP not set, using default 8" << endl;
    }
    
    string conc_str = get_env_var("CONC");
    if (!conc_str.empty()) {
        cfg.conc = stoi(conc_str);
    } else {
        cout << "WARNING: CONC not set, using default 8" << endl;
    }
    
    string isl_str = get_env_var("ISL");
    if (!isl_str.empty()) {
        cfg.isl = stoi(isl_str);
    } else {
        cout << "WARNING: ISL not set, using default 1024" << endl;
    }
    
    string osl_str = get_env_var("OSL");
    if (!osl_str.empty()) {
        cfg.osl = stoi(osl_str);
    } else {
        cout << "WARNING: OSL not set, using default 1024" << endl;
    }
    
    string max_model_len_str = get_env_var("MAX_MODEL_LEN");
    if (!max_model_len_str.empty()) {
        cfg.max_model_len = stoi(max_model_len_str);
    } else {
        cout << "WARNING: MAX_MODEL_LEN not set, using default 16384" << endl;
    }
    
    string ratio_str = get_env_var("RANDOM_RANGE_RATIO");
    if (!ratio_str.empty()) {
        cfg.random_range_ratio = stod(ratio_str);
    } else {
        cout << "WARNING: RANDOM_RANGE_RATIO not set, using default 1.0" << endl;
    }
    
    cfg.result_filename = get_env_var("RESULT_FILENAME", "result");
    
    string num_prompts_str = get_env_var("NUM_PROMPTS");
    if (!num_prompts_str.empty()) {
        cfg.num_prompts = stoi(num_prompts_str);
    } else {
        cout << "WARNING: NUM_PROMPTS not set, calculating based on CONC (GPT-OSS: CONC * 10)" << endl;
        cfg.num_prompts = cfg.conc * 10;
    }
    
    cfg.lb_url_override = get_env_var("LB_URL_OVERRIDE");
    
    cout << "============================================" << endl;
    cout << "Configuration:" << endl;
    cout << "============================================" << endl;
    cout << "MODEL:        " << cfg.model << endl;
    cout << "PORT:         " << cfg.port << endl;
    cout << "TP:           " << cfg.tp << endl;
    cout << "CONC:         " << cfg.conc << endl;
    cout << "ISL:          " << cfg.isl << endl;
    cout << "OSL:          " << cfg.osl << endl;
    cout << "MAX_MODEL_LEN: " << cfg.max_model_len << endl;
    cout << "NUM_PROMPTS:  " << cfg.num_prompts << endl;
    cout << "RESULT_FILE:  " << cfg.result_filename << ".json" << endl;
    cout << "============================================" << endl;
    cout << endl;
    
    AccuracyMetrics acc_metrics;
    if (run_accuracy_test(cfg, acc_metrics) != 0) {
        return 1;
    }
    
    if (validate_accuracy(acc_metrics) != 0) {
        return 1;
    }
    
    return run_single_test(cfg, acc_metrics);
}

