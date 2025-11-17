#ifndef GENETIC_ODE_SOLVER_H
#define GENETIC_ODE_SOLVER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct GeneticProgram GeneticProgram;
typedef struct HashTable HashTable;
typedef struct CascadingHashTables CascadingHashTables;
typedef struct GeneticSolver GeneticSolver;

// Genetic Program Node Types
typedef enum {
    GP_NODE_CONSTANT,      // Constant value
    GP_NODE_VARIABLE,      // Variable (t, x, y, etc.)
    GP_NODE_ADD,           // Addition
    GP_NODE_SUB,           // Subtraction
    GP_NODE_MUL,           // Multiplication
    GP_NODE_DIV,           // Division
    GP_NODE_POW,           // Power
    GP_NODE_SIN,           // Sine
    GP_NODE_COS,           // Cosine
    GP_NODE_EXP,           // Exponential
    GP_NODE_LOG,           // Natural logarithm
    GP_NODE_SQRT           // Square root
} GPNodeType;

// Genetic Program Node
typedef struct GPNode {
    GPNodeType type;
    double value;          // For constants
    int var_index;         // For variables (0=t, 1=x, 2=y, etc.)
    struct GPNode* left;
    struct GPNode* right;
} GPNode;

// Genetic Program Structure
struct GeneticProgram {
    GPNode* root;
    double fitness;
    double* parameters;    // Evolved parameters
    size_t param_count;
    size_t complexity;     // Tree size/complexity
};

// Hash Table Entry for Cached Solutions
typedef struct HashEntry {
    uint64_t key_hash;     // Hash of problem signature
    GeneticProgram* solution;
    double* cached_values; // Cached evaluation results
    size_t cache_size;
    struct HashEntry* next; // For collision chaining
} HashEntry;

// Single-Level Hash Table
struct HashTable {
    HashEntry** buckets;
    size_t bucket_count;
    size_t entry_count;
    double load_factor;
};

// Cascading Hash Tables (Multi-Level)
struct CascadingHashTables {
    HashTable** levels;
    size_t level_count;
    size_t* level_sizes;   // Size of each level
    double* similarity_thresholds; // Thresholds for each level
};

// ODE/PDE Problem Definition
typedef struct ProblemDef {
    int problem_type;      // 0=ODE, 1=PDE
    int dimension;          // State dimension
    int spatial_dim;       // For PDEs: 1D, 2D, 3D
    double* initial_conditions;
    double* domain_bounds;  // [t_min, t_max] for ODE, [x_min, x_max, y_min, y_max, ...] for PDE
    double tolerance;
    size_t max_steps;
} ProblemDef;

// Genetic Algorithm Parameters
typedef struct GAParams {
    size_t population_size;
    size_t max_generations;
    double mutation_rate;
    double crossover_rate;
    double selection_pressure;
    size_t tournament_size;
    size_t elitism_count;
    double min_fitness;
    bool use_parallel;
} GAParams;

// Fitness Evaluation Function
typedef double (*FitnessFunction)(GeneticProgram* program, ProblemDef* problem, void* user_data);

// Genetic Solver Structure
struct GeneticSolver {
    // Problem definition
    ProblemDef problem;
    
    // Genetic algorithm parameters
    GAParams ga_params;
    
    // Population
    GeneticProgram** population;
    size_t population_size;
    
    // Cascading hash tables for caching
    CascadingHashTables* hash_tables;
    
    // Fitness function
    FitnessFunction fitness_func;
    void* fitness_user_data;
    
    // Statistics
    size_t generation;
    double best_fitness;
    GeneticProgram* best_solution;
    size_t hash_hits;
    size_t hash_misses;
    size_t evaluations;
    
    // Offline mode
    bool offline_mode;
    char* cache_file;
};

// ============================================================================
// Hash Table Functions
// ============================================================================

/**
 * Create a new hash table
 * @param bucket_count Number of buckets
 * @return New hash table or NULL on error
 */
HashTable* hash_table_create(size_t bucket_count);

/**
 * Free a hash table
 */
void hash_table_free(HashTable* table);

/**
 * Compute hash for a problem signature
 * @param problem Problem definition
 * @param param_hash Hash of parameters
 * @return 64-bit hash value
 */
uint64_t hash_problem_signature(ProblemDef* problem, uint64_t param_hash);

/**
 * Insert solution into hash table
 * @param table Hash table
 * @param key_hash Hash key
 * @param solution Genetic program solution
 * @return true on success
 */
bool hash_table_insert(HashTable* table, uint64_t key_hash, GeneticProgram* solution);

/**
 * Lookup solution in hash table
 * @param table Hash table
 * @param key_hash Hash key
 * @return Genetic program solution or NULL if not found
 */
GeneticProgram* hash_table_lookup(HashTable* table, uint64_t key_hash);

/**
 * Remove entry from hash table
 */
bool hash_table_remove(HashTable* table, uint64_t key_hash);

// ============================================================================
// Cascading Hash Tables Functions
// ============================================================================

/**
 * Create cascading hash tables
 * @param level_count Number of levels
 * @param level_sizes Array of bucket counts for each level
 * @param thresholds Similarity thresholds for each level
 * @return New cascading hash tables or NULL on error
 */
CascadingHashTables* cascading_hash_create(size_t level_count, 
                                           size_t* level_sizes, 
                                           double* thresholds);

/**
 * Free cascading hash tables
 */
void cascading_hash_free(CascadingHashTables* cascading);

/**
 * Insert solution into cascading hash tables
 * @param cascading Cascading hash tables
 * @param key_hash Hash key
 * @param solution Genetic program solution
 * @param similarity Similarity score (0.0 to 1.0)
 * @return Level at which inserted, or -1 on error
 */
int cascading_hash_insert(CascadingHashTables* cascading, 
                         uint64_t key_hash, 
                         GeneticProgram* solution, 
                         double similarity);

/**
 * Lookup solution in cascading hash tables (tries all levels)
 * @param cascading Cascading hash tables
 * @param key_hash Hash key
 * @param min_similarity Minimum similarity threshold
 * @return Genetic program solution or NULL if not found
 */
GeneticProgram* cascading_hash_lookup(CascadingHashTables* cascading, 
                                     uint64_t key_hash, 
                                     double min_similarity);

/**
 * Get statistics from cascading hash tables
 */
void cascading_hash_stats(CascadingHashTables* cascading, 
                         size_t* total_entries, 
                         size_t* entries_per_level);

// ============================================================================
// Genetic Program Functions
// ============================================================================

/**
 * Create a new genetic program node
 */
GPNode* gp_node_create(GPNodeType type, double value, int var_index);

/**
 * Free a genetic program node and its subtree
 */
void gp_node_free(GPNode* node);

/**
 * Copy a genetic program (deep copy)
 */
GeneticProgram* gp_copy(GeneticProgram* program);

/**
 * Free a genetic program
 */
void gp_free(GeneticProgram* program);

/**
 * Evaluate genetic program at given point
 * @param program Genetic program
 * @param vars Variable values (t, x, y, ...)
 * @param var_count Number of variables
 * @return Evaluated value
 */
double gp_evaluate(GeneticProgram* program, double* vars, size_t var_count);

/**
 * Compute program complexity (tree size)
 */
size_t gp_complexity(GeneticProgram* program);

/**
 * Mutate a genetic program
 * @param program Program to mutate
 * @param mutation_rate Mutation probability
 * @return Mutated program (may be same or new)
 */
GeneticProgram* gp_mutate(GeneticProgram* program, double mutation_rate);

/**
 * Crossover two genetic programs
 * @param parent1 First parent
 * @param parent2 Second parent
 * @param child1 Output: first child
 * @param child2 Output: second child
 */
void gp_crossover(GeneticProgram* parent1, 
                  GeneticProgram* parent2, 
                  GeneticProgram** child1, 
                  GeneticProgram** child2);

/**
 * Generate random genetic program
 * @param max_depth Maximum tree depth
 * @param var_count Number of variables
 * @return Random program
 */
GeneticProgram* gp_random(size_t max_depth, size_t var_count);

// ============================================================================
// Genetic Solver Functions
// ============================================================================

/**
 * Create a new genetic solver
 * @param problem Problem definition
 * @param ga_params Genetic algorithm parameters
 * @param fitness_func Fitness evaluation function
 * @param user_data User data for fitness function
 * @return New genetic solver or NULL on error
 */
GeneticSolver* genetic_solver_create(ProblemDef* problem, 
                                     GAParams* ga_params,
                                     FitnessFunction fitness_func,
                                     void* user_data);

/**
 * Free genetic solver
 */
void genetic_solver_free(GeneticSolver* solver);

/**
 * Initialize population
 */
bool genetic_solver_init_population(GeneticSolver* solver);

/**
 * Run one generation of genetic algorithm
 * @return true if converged
 */
bool genetic_solver_evolve_generation(GeneticSolver* solver);

/**
 * Run full genetic algorithm until convergence or max generations
 * @return Best solution found
 */
GeneticProgram* genetic_solver_solve(GeneticSolver* solver);

/**
 * Enable offline mode with cache file
 */
bool genetic_solver_set_offline_mode(GeneticSolver* solver, const char* cache_file);

/**
 * Load cached solutions from file
 */
bool genetic_solver_load_cache(GeneticSolver* solver, const char* cache_file);

/**
 * Save solutions to cache file
 */
bool genetic_solver_save_cache(GeneticSolver* solver, const char* cache_file);

/**
 * Get solver statistics
 */
void genetic_solver_get_stats(GeneticSolver* solver, 
                             size_t* generation,
                             double* best_fitness,
                             size_t* hash_hits,
                             size_t* hash_misses,
                             size_t* evaluations);

/**
 * Set cascading hash table parameters
 */
bool genetic_solver_set_hash_tables(GeneticSolver* solver,
                                   size_t level_count,
                                   size_t* level_sizes,
                                   double* thresholds);

#ifdef __cplusplus
}
#endif

#endif // GENETIC_ODE_SOLVER_H
