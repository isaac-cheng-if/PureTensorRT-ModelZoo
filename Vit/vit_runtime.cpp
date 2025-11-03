/*
 * ViT Runtime Inference
 * Handles engine deserialization, inference execution, and post-processing
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <cassert>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// ==================== Configuration ====================
namespace Config {
    const int BATCH_SIZE = 1;
    const int NUM_CLASS = 1000;
    const int INPUT_H = 224;
    const int INPUT_W = 224;
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output";
    const int GPU_ID = 0;
}

// ==================== CUDA Error Checking ====================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ==================== TensorRT Logger ====================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// ==================== Performance Statistics ====================
struct PerfStats {
    int iterations;
    double avg_time;
    double min_time;
    double max_time;
    double total_time;
    double throughput;

    PerfStats() : iterations(0), avg_time(0), min_time(0), max_time(0),
                  total_time(0), throughput(0) {}

    void print(const std::string& title) const {
        std::cout << "\n=== " << title << " (GPU Only) ===\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Average GPU time: " << std::fixed << std::setprecision(3) << avg_time << " ms\n";
        std::cout << "Min GPU time: " << std::fixed << std::setprecision(3) << min_time << " ms\n";
        std::cout << "Max GPU time: " << std::fixed << std::setprecision(3) << max_time << " ms\n";
        std::cout << "Total GPU time: " << std::fixed << std::setprecision(3) << total_time << " ms\n";
        std::cout << "GPU Throughput: " << std::fixed << std::setprecision(2) << throughput << " FPS\n";
        std::cout << "=====================================\n";
    }
};

// ==================== ImageNet Class Names ====================
static const std::vector<std::string> CLASS_NAMES = {
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray",
    "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee", "water ouzel", "kite",
    "bald eagle", "vulture", "great grey owl", "European fire salamander", "common newt",
    "eft", "spotted salamander", "axolotl", "bullfrog", "tree frog", "tailed frog",
    "loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle", "banded gecko",
    "common iguana", "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
    "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake",
    "green snake", "king snake", "garter snake", "water snake", "vine snake", "night snake",
    "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake", "horned viper",
    "diamondback", "sidewinder", "trilobite", "harvestman", "scorpion", "black and gold garden spider",
    "barn spider", "garden spider", "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge",
    "African grey", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater",
    "hornbill", "hummingbird", "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
    "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish",
    "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug",
    "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "king crab",
    "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork",
    "black stork", "spoonbill", "flamingo", "little blue heron", "American egret", "bittern",
    "crane", "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
    "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican", "king penguin",
    "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese spaniel",
    "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier",
    "Rhodesian ridgeback", "Afghan hound", "basset", "beagle", "bloodhound", "bluetick",
    "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
    "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
    "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
    "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier",
    "Irish terrier", "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier",
    "Lakeland terrier", "Sealyham terrier", "Airedale", "cairn", "Australian terrier",
    "Dandie Dinmont", "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer",
    "Scotch terrier", "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier",
    "West Highland white terrier", "Lhasa", "flat-coated retriever", "curly-coated retriever",
    "golden retriever", "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer",
    "vizsla", "English setter", "Irish setter", "Gordon setter", "Brittany spaniel", "clumber",
    "English springer", "Welsh springer spaniel", "cocker spaniel", "Sussex spaniel", "Irish water spaniel",
    "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", "komondor",
    "Old English sheepdog", "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres",
    "Rottweiler", "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog",
    "Bernese mountain dog", "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff",
    "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute", "Siberian husky",
    "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed",
    "Pomeranian", "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "toy poodle",
    "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf", "white wolf",
    "red wolf", "coyote", "dingo", "African hunting dog", "hyena", "red fox", "kit fox",
    "Arctic fox", "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat",
    "lion", "tiger", "jaguar", "leopard", "snow leopard", "lynx", "bobcat", "leopard cat",
    "cougar", "lynx", "cheetah", "brown bear", "American black bear", "ice bear", "sloth bear",
    "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "long-horned beetle",
    "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket", "walking stick", "cockroach", "mantis", "cicada", "leafhopper", "lacewing",
    "dragonfly", "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly", "sulphur butterfly",
    "lycaenid", "starfish", "sea urchin", "sea cucumber", "wood rabbit", "hare", "Angora",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel",
    "zebra", "hog", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram", "bighorn", "ibex", "hartebeest", "impala", "gazelle", "Arabian camel", "llama",
    "weasel", "mink", "polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo",
    "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon",
    "patas", "baboon", "macaque", "langur", "colobus", "proboscis monkey", "marmoset",
    "capuchin", "howler monkey", "titi", "spider monkey", "squirrel monkey", "Madagascar cat",
    "indri", "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta",
    "eel", "coho", "rock beauty", "anemone fish", "sturgeon", "gar", "lionfish", "puffer",
    "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier",
    "airliner", "airship", "altar", "ambulance", "amphibian", "analog clock", "apiary",
    "apron", "ashcan", "assault rifle", "backpack", "bakery", "balance beam", "balloon",
    "ballpoint", "Band Aid", "banjo", "bannister", "barbell", "barber chair", "barbershop",
    "barn", "barometer", "barrel", "barrow", "baseball", "basketball", "bassinet", "bassoon",
    "bathing cap", "bath towel", "bathtub", "beach wagon", "beacon", "beaker", "bearskin",
    "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two", "bikini",
    "binder", "binoculars", "birdhouse", "boathouse", "bobsled", "bolo tie", "bonnet",
    "bookcase", "bookshop", "bottlecap", "bow", "bow tie", "brass", "brassiere", "breakwater",
    "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "bullet train", "butcher shop",
    "cab", "caldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror",
    "carousel", "carpenter's kit", "carriage", "carrier", "carton", "car wheel", "cash machine",
    "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "cellular telephone",
    "chain", "chainlink fence", "chain mail", "chain saw", "chest", "chiffonier", "chime",
    "china cabinet", "Christmas stocking", "church", "cinema", "cleaver", "cliff dwelling",
    "cloak", "clog", "cocktail shaker", "coffee mug", "coffeepot", "coil", "combination lock",
    "computer keyboard", "confectionery", "container ship", "convertible", "corkscrew", "cornet",
    "cowboy boot", "cowboy hat", "cradle", "crane", "crash helmet", "crate", "crib", "Crock Pot",
    "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "dial telephone",
    "diaper", "digital clock", "digital watch", "dining table", "dishrag", "dishwasher",
    "disk brake", "dock", "dogsled", "dome", "doormat", "drilling platform", "drum",
    "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive",
    "entertainment center", "envelope", "espresso maker", "face powder", "feather boa",
    "file", "fireboat", "fire engine", "fire screen", "flagpole", "flute", "folding chair",
    "football helmet", "forklift", "fountain", "fountain pen", "four-poster", "freight car",
    "French horn", "frying pan", "fur coat", "garbage truck", "gasmask", "gas pump", "goblet",
    "go-kart", "golf ball", "golfcart", "gondola", "gong", "gown", "grand piano", "greenhouse",
    "grille", "grocery store", "guillotine", "hair slide", "hair spray", "half track",
    "hammer", "hamper", "hand blower", "hand-held computer", "handkerchief", "hard disc",
    "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb",
    "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass", "iPod", "iron",
    "jack-o'-lantern", "jean", "jeep", "jersey", "jigsaw puzzle", "jinrikisha", "joystick",
    "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop", "lawn mower",
    "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "liner",
    "lipstick", "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill", "magnetic compass",
    "mailbag", "mailbox", "maillot", "manhole cover", "maraca", "marimba", "mask", "matchstick",
    "maypole", "maze", "measuring cup", "medicine chest", "megalith", "microphone", "microwave",
    "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten",
    "mixing bowl", "mobile home", "Model T", "modem", "monastery", "monitor", "moped",
    "mortar", "mortarboard", "mosque", "mosquito net", "motor scooter", "mountain bike",
    "mountain tent", "mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace",
    "necklace", "nipple", "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil filter",
    "organ", "oscilloscope", "overskirt", "oxcart", "oxygen mask", "packet", "paddle",
    "paddlewheel", "padlock", "paintbrush", "pajama", "palace", "panpipe", "paper towel",
    "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio",
    "pay-phone", "pedestal", "pencil box", "pencil sharpener", "perfume", "Petri dish",
    "photocopier", "pick", "pickelhaube", "picket fence", "pickup", "pier", "piggy bank",
    "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate", "pitcher", "plane",
    "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera",
    "pole", "police van", "poncho", "pool table", "pop bottle", "pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "puck", "punching bag", "purse",
    "quill", "quilt", "racer", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control",
    "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "rubber eraser",
    "rugby ball", "rule", "running shoe", "safe", "safety pin", "saltshaker", "sandal",
    "sarong", "sax", "scabbard", "scale", "school bus", "schooner", "scoreboard", "screen",
    "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe shop", "shoji",
    "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski",
    "ski mask", "sleeping bag", "slide rule", "sliding door", "slot", "snorkel", "snowmobile",
    "snowplow", "soap dispenser", "soccer ball", "sock", "solar dish", "sombrero", "soup bowl",
    "space bar", "space heater", "space shuttle", "spatula", "speedboat", "spider web",
    "spindle", "sports car", "spotlight", "stage", "steam locomotive", "steel arch bridge",
    "steel drum", "stethoscope", "stole", "stone wall", "stopwatch", "stove", "strainer",
    "streetcar", "stretcher", "studio couch", "stupa", "submarine", "suit", "sundial",
    "sunglass", "sunglasses", "sunscreen", "suspension bridge", "swab", "sweatshirt",
    "swimming trunks", "swing", "switch", "syringe", "table lamp", "tank", "tape player",
    "teapot", "teddy", "television", "tennis ball", "thatch", "theater curtain", "thimble",
    "thresher", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch",
    "totem pole", "tow truck", "toyshop", "tractor", "trailer truck", "tray", "trench coat",
    "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub",
    "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright", "vacuum",
    "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "warplane", "washbasin", "washer",
    "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "wig", "window screen",
    "window shade", "Windsor tie", "wine bottle", "wing", "wok", "wooden spoon", "wool",
    "worm fence", "wreck", "yawl", "yurt", "web site", "comic book", "crossword puzzle",
    "street sign", "traffic light", "book jacket", "menu", "plate", "guacamole", "consomme",
    "hot pot", "trifle", "ice cream", "ice lolly", "French loaf", "bagel", "pretzel",
    "cheeseburger", "hotdog", "mashed potato", "head cabbage", "broccoli", "cauliflower",
    "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon",
    "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay",
    "carbonara", "chocolate sauce", "dough", "meat loaf", "pizza", "potpie", "burrito",
    "red wine", "espresso", "cup", "eggnog", "alp", "bubble", "cliff", "coral reef",
    "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley", "volcano",
    "ballplayer", "groom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper",
    "corn", "acorn", "hip", "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn",
    "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue"
};

// ==================== Utility Functions ====================
int ReadFilesInDir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            if (cur_file_name.find(".jpg") != std::string::npos ||
                cur_file_name.find(".jpeg") != std::string::npos ||
                cur_file_name.find(".png") != std::string::npos) {
                file_names.push_back(cur_file_name);
            }
        }
    }
    closedir(p_dir);
    return 0;
}

// ==================== Image Preprocessing ====================
cv::Mat PreProcessImage(const cv::Mat& img, bool debug = false) {
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(Config::INPUT_W, Config::INPUT_H));
    
    if (debug) {
        std::cout << "Resized image shape: " << resized_img.size() << std::endl;
    }
    
    return resized_img;
}

// ==================== ViT Runtime Class ====================
class ViTRuntime {
private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    
    void* gpu_input_buffer;
    void* gpu_output_buffer;
    void* cpu_output_buffer;
    
    int input_size;
    int output_size;
    
    cudaStream_t stream;

public:
    ViTRuntime() : runtime(nullptr), engine(nullptr), context(nullptr),
                   gpu_input_buffer(nullptr), gpu_output_buffer(nullptr), cpu_output_buffer(nullptr),
                   input_size(0), output_size(0), stream(nullptr) {}

    ~ViTRuntime() {
        if (stream) cudaStreamDestroy(stream);
        if (cpu_output_buffer) free(cpu_output_buffer);
        if (gpu_output_buffer) cudaFree(gpu_output_buffer);
        if (gpu_input_buffer) cudaFree(gpu_input_buffer);
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
    }

    bool Deserialize(const std::string& engine_file) {
        std::cout << "Loading engine: " << engine_file << std::endl;
        
        std::ifstream file(engine_file, std::ios::binary);
        if (!file.good()) {
            std::cerr << "ERROR: Unable to read engine file: " << engine_file << std::endl;
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "ERROR: createInferRuntime failed" << std::endl;
            return false;
        }

        engine = runtime->deserializeCudaEngine(engine_data.data(), size);
        if (!engine) {
            std::cerr << "ERROR: deserializeCudaEngine failed" << std::endl;
            return false;
        }

        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "ERROR: createExecutionContext failed" << std::endl;
            return false;
        }

        // Allocate GPU memory
        input_size = Config::BATCH_SIZE * 3 * Config::INPUT_H * Config::INPUT_W * sizeof(float);
        output_size = Config::BATCH_SIZE * Config::NUM_CLASS * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&gpu_input_buffer, input_size));
        CUDA_CHECK(cudaMalloc(&gpu_output_buffer, output_size));
        cpu_output_buffer = malloc(output_size);

        CUDA_CHECK(cudaStreamCreate(&stream));

        std::cout << "Engine loaded successfully!" << std::endl;
        std::cout << "Input size: " << input_size << " bytes" << std::endl;
        std::cout << "Output size: " << output_size << " bytes" << std::endl;
        
        return true;
    }

    bool Infer(const cv::Mat& img, std::vector<float>& output) {
        // Preprocess image
        cv::Mat processed_img = PreProcessImage(img);
        
        // Convert BGR to RGB and normalize
        cv::Mat rgb_img;
        cv::cvtColor(processed_img, rgb_img, cv::COLOR_BGR2RGB);
        rgb_img.convertTo(rgb_img, CV_32F, 1.0/255.0);
        
        // Normalize with ImageNet mean and std
        float mean[] = {0.485, 0.456, 0.406};
        float std[] = {0.229, 0.224, 0.225};
        
        std::vector<cv::Mat> channels;
        cv::split(rgb_img, channels);
        
        for (int i = 0; i < 3; i++) {
            channels[i] = (channels[i] - mean[i]) / std[i];
        }
        
        cv::Mat normalized_img;
        cv::merge(channels, normalized_img);
        
        // Convert to tensor format (HWC -> CHW)
        // Manually reorder from HWC to CHW format
        std::vector<float> input_data(input_size / sizeof(float));
        int img_size = Config::INPUT_H * Config::INPUT_W;
        
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < Config::INPUT_H; h++) {
                for (int w = 0; w < Config::INPUT_W; w++) {
                    input_data[c * img_size + h * Config::INPUT_W + w] = 
                        channels[c].at<float>(h, w);
                }
            }
        }
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpyAsync(gpu_input_buffer, input_data.data(), input_size, 
                                   cudaMemcpyHostToDevice, stream));
        
        // Run inference
        void* bindings[] = {gpu_input_buffer, gpu_output_buffer};
        bool status = context->executeV2(bindings);
        if (!status) {
            std::cerr << "ERROR: executeV2 failed" << std::endl;
            return false;
        }
        
        // Copy result back to CPU
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_output_buffer, output_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Convert to vector
        float* output_ptr = static_cast<float*>(cpu_output_buffer);
        output.assign(output_ptr, output_ptr + Config::NUM_CLASS);
        
        return true;
    }

    PerfStats PerformanceTest(const cv::Mat& img, int iterations = 100) {
        PerfStats stats;
        stats.iterations = iterations;
        
        std::vector<double> times;
        times.reserve(iterations);
        
        // Warmup
        std::vector<float> dummy_output;
        for (int i = 0; i < 10; i++) {
            Infer(img, dummy_output);
        }
        
        // GPU throughput test - measure only GPU execution time
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Create CUDA events for precise GPU timing
        cudaEvent_t start_event, end_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&end_event));
        
        // Performance test - measure GPU execution time only
        for (int i = 0; i < iterations; i++) {
            // Preprocess image (CPU)
            cv::Mat processed_img = PreProcessImage(img);
            cv::Mat rgb_img;
            cv::cvtColor(processed_img, rgb_img, cv::COLOR_BGR2RGB);
            rgb_img.convertTo(rgb_img, CV_32F, 1.0/255.0);
            
            float mean[] = {0.485, 0.456, 0.406};
            float std[] = {0.229, 0.224, 0.225};
            std::vector<cv::Mat> channels;
            cv::split(rgb_img, channels);
            for (int c = 0; c < 3; c++) {
                channels[c] = (channels[c] - mean[c]) / std[c];
            }
            
            std::vector<float> input_data(input_size / sizeof(float));
            int img_size = Config::INPUT_H * Config::INPUT_W;
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < Config::INPUT_H; h++) {
                    for (int w = 0; w < Config::INPUT_W; w++) {
                        input_data[c * img_size + h * Config::INPUT_W + w] = 
                            channels[c].at<float>(h, w);
                    }
                }
            }
            
            // Copy to GPU
            CUDA_CHECK(cudaMemcpyAsync(gpu_input_buffer, input_data.data(), input_size, 
                                       cudaMemcpyHostToDevice, stream));
            
            // Start GPU timing
            CUDA_CHECK(cudaEventRecord(start_event, stream));
            
            // Run inference
            void* bindings[] = {gpu_input_buffer, gpu_output_buffer};
            bool status = context->executeV2(bindings);
            if (!status) {
                std::cerr << "ERROR: executeV2 failed" << std::endl;
                return stats;
            }
            
            // End GPU timing
            CUDA_CHECK(cudaEventRecord(end_event, stream));
            CUDA_CHECK(cudaEventSynchronize(end_event));
            
            // Calculate GPU execution time
            float gpu_time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start_event, end_event));
            times.push_back(gpu_time_ms);
            
            // Copy result back to CPU (not timed)
            CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_output_buffer, output_size,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        
        // Clean up CUDA events
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(end_event));
        
        // Calculate statistics
        stats.total_time = std::accumulate(times.begin(), times.end(), 0.0);
        stats.avg_time = stats.total_time / iterations;
        stats.min_time = *std::min_element(times.begin(), times.end());
        stats.max_time = *std::max_element(times.begin(), times.end());
        stats.throughput = 1000.0 / stats.avg_time; // FPS
        
        return stats;
    }
};

// ==================== Main Function ====================
int main(int argc, char** argv) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ViT Runtime Inference" << std::endl;
    std::cout << "TensorRT Engine Inference" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;

    if (argc < 3) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./vit_runtime <engine_file> <image_path_or_dir>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  ./vit_runtime vit_base.engine kitten.jpg" << std::endl;
        std::cout << "  ./vit_runtime vit_base.engine ./images/" << std::endl;
        return -1;
    }

    cudaSetDevice(Config::GPU_ID);

    ViTRuntime runtime;
    if (!runtime.Deserialize(argv[1])) {
        std::cerr << "Failed to load engine" << std::endl;
        return -1;
    }

    std::string input_path = argv[2];
    std::vector<std::string> image_files;

    // Check if input is a directory or single file
    if (input_path.back() == '/') {
        // Directory
        if (ReadFilesInDir(input_path.c_str(), image_files) != 0) {
            std::cerr << "ERROR: Unable to read directory: " << input_path << std::endl;
            return -1;
        }
        
        std::cout << "Found " << image_files.size() << " images in directory" << std::endl;
        
        // Process all images
        for (const auto& img_file : image_files) {
            std::string full_path = input_path + img_file;
            cv::Mat img = cv::imread(full_path);
            if (img.empty()) {
                std::cerr << "ERROR: Unable to load image: " << full_path << std::endl;
                continue;
            }
            
            std::cout << "\n=== Processing: " << img_file << " ===" << std::endl;
            
            std::vector<float> output;
            if (runtime.Infer(img, output)) {
                // Find top-5 predictions
                std::vector<std::pair<float, int>> predictions;
                for (int i = 0; i < Config::NUM_CLASS; i++) {
                    predictions.emplace_back(output[i], i);
                }
                
                std::sort(predictions.rbegin(), predictions.rend());
                
            // Apply softmax to convert logits to probabilities
            std::vector<float> probabilities(Config::NUM_CLASS);
            
            // Find max logit for numerical stability
            float max_logit = *std::max_element(output.begin(), output.end());
            
            // Calculate softmax
            double sum_exp = 0.0;
            for (int i = 0; i < Config::NUM_CLASS; i++) {
                float exp_val = std::exp(output[i] - max_logit);
                probabilities[i] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize probabilities
            for (int i = 0; i < Config::NUM_CLASS; i++) {
                probabilities[i] /= sum_exp;
            }
            
            // Find top-5 probabilities
            std::vector<std::pair<float, int>> prob_predictions;
            for (int i = 0; i < Config::NUM_CLASS; i++) {
                prob_predictions.emplace_back(probabilities[i], i);
            }
            
            std::sort(prob_predictions.rbegin(), prob_predictions.rend());
            
            std::cout << "Top 5 predictions (with softmax):" << std::endl;
            for (int i = 0; i < 5; i++) {
                int class_id = prob_predictions[i].second;
                float probability = prob_predictions[i].first;
                std::string class_name = (class_id < CLASS_NAMES.size()) ? 
                                        CLASS_NAMES[class_id] : "Unknown";
                std::cout << "  " << (i+1) << ". " << class_name 
                          << " (ID: " << class_id << ") - " 
                          << std::fixed << std::setprecision(6) << probability 
                          << " (" << std::fixed << std::setprecision(2) << (probability * 100) << "%)" << std::endl;
            }
                
                // GPU Throughput test
                auto perf_stats = runtime.PerformanceTest(img, 100);
                perf_stats.print("GPU Throughput Test");
            }
        }
    } else {
        // Single file
        cv::Mat img = cv::imread(input_path);
        if (img.empty()) {
            std::cerr << "ERROR: Unable to load image: " << input_path << std::endl;
            return -1;
        }
        
        std::cout << "Processing single image: " << input_path << std::endl;
        
        std::vector<float> output;
        if (runtime.Infer(img, output)) {
            // Find top-5 predictions
            std::vector<std::pair<float, int>> predictions;
            for (int i = 0; i < Config::NUM_CLASS; i++) {
                predictions.emplace_back(output[i], i);
            }
            
            std::sort(predictions.rbegin(), predictions.rend());
            
            // Apply softmax to convert logits to probabilities
            std::vector<float> probabilities(Config::NUM_CLASS);
            
            // Find max logit for numerical stability
            float max_logit = *std::max_element(output.begin(), output.end());
            
            // Calculate softmax
            double sum_exp = 0.0;
            for (int i = 0; i < Config::NUM_CLASS; i++) {
                float exp_val = std::exp(output[i] - max_logit);
                probabilities[i] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize probabilities
            for (int i = 0; i < Config::NUM_CLASS; i++) {
                probabilities[i] /= sum_exp;
            }
            
            // Find top-5 probabilities
            std::vector<std::pair<float, int>> prob_predictions;
            for (int i = 0; i < Config::NUM_CLASS; i++) {
                prob_predictions.emplace_back(probabilities[i], i);
            }
            
            std::sort(prob_predictions.rbegin(), prob_predictions.rend());
            
            std::cout << "Top 5 predictions (with softmax):" << std::endl;
            for (int i = 0; i < 5; i++) {
                int class_id = prob_predictions[i].second;
                float probability = prob_predictions[i].first;
                std::string class_name = (class_id < CLASS_NAMES.size()) ? 
                                        CLASS_NAMES[class_id] : "Unknown";
                std::cout << "  " << (i+1) << ". " << class_name 
                          << " (ID: " << class_id << ") - " 
                          << std::fixed << std::setprecision(6) << probability 
                          << " (" << std::fixed << std::setprecision(2) << (probability * 100) << "%)" << std::endl;
            }
            
            // GPU Throughput test
            auto perf_stats = runtime.PerformanceTest(img, 100);
            perf_stats.print("GPU Throughput Test");
        }
    }

    std::cout << "\nInference completed!" << std::endl;
    return 0;
}
