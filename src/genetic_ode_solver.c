#include "genetic_ode_solver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

// ============================================================================
// Hash Table Implementation
// ============================================================================

HashTable* hash_table_create(size_t bucket_count) {
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    if (!table) return NULL;
    
    table->buckets = (HashEntry**)calloc(bucket_count, sizeof(HashEntry*));
    if (!table->buckets) {
        free(table);
        return NULL;
    }
    
    table->bucket_count = bucket_count;
    table->entry_count = 0;
    table->load_factor = 0.0;
    
    return table;
}

void hash_table_free(HashTable* table) {
    if (!table) return;
    
    for (size_t i = 0; i < table->bucket_count; i++) {
        HashEntry* entry = table->buckets[i];
        while (entry) {
            HashEntry* next = entry->next;
            // Note: We don't free the solution here as it may be shared
            if (entry->cached_values) {
                free(entry->cached_values);
            }
            free(entry);
            entry = next;
        }
    }
    
    free(table->buckets);
    free(table);
}

// FNV-1a hash function
static uint64_t fnv1a_hash(const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t hash = 14695981039346656037ULL;
    
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    
    return hash;
}

uint64_t hash_problem_signature(ProblemDef* problem, uint64_t param_hash) {
    uint64_t hash = param_hash;
    
    // Hash problem type and dimensions
    hash ^= fnv1a_hash(&problem->problem_type, sizeof(int));
    hash ^= fnv1a_hash(&problem->dimension, sizeof(int));
    hash ^= fnv1a_hash(&problem->spatial_dim, sizeof(int));
    
    // Hash initial conditions
    if (problem->initial_conditions) {
        hash ^= fnv1a_hash(problem->initial_conditions, 
                          problem->dimension * sizeof(double));
    }
    
    // Hash domain bounds
    if (problem->domain_bounds) {
        size_t bounds_size = (problem->problem_type == 0) ? 2 : 
                            (2 + 2 * problem->spatial_dim);
        hash ^= fnv1a_hash(problem->domain_bounds, bounds_size * sizeof(double));
    }
    
    return hash;
}

bool hash_table_insert(HashTable* table, uint64_t key_hash, GeneticProgram* solution) {
    if (!table || !solution) return false;
    
    size_t bucket_idx = key_hash % table->bucket_count;
    HashEntry* entry = table->buckets[bucket_idx];
    
    // Check if key already exists
    while (entry) {
        if (entry->key_hash == key_hash) {
            // Update existing entry
            entry->solution = solution;
            return true;
        }
        entry = entry->next;
    }
    
    // Create new entry
    entry = (HashEntry*)malloc(sizeof(HashEntry));
    if (!entry) return false;
    
    entry->key_hash = key_hash;
    entry->solution = solution;
    entry->cached_values = NULL;
    entry->cache_size = 0;
    entry->next = table->buckets[bucket_idx];
    table->buckets[bucket_idx] = entry;
    
    table->entry_count++;
    table->load_factor = (double)table->entry_count / table->bucket_count;
    
    return true;
}

GeneticProgram* hash_table_lookup(HashTable* table, uint64_t key_hash) {
    if (!table) return NULL;
    
    size_t bucket_idx = key_hash % table->bucket_count;
    HashEntry* entry = table->buckets[bucket_idx];
    
    while (entry) {
        if (entry->key_hash == key_hash) {
            return entry->solution;
        }
        entry = entry->next;
    }
    
    return NULL;
}

bool hash_table_remove(HashTable* table, uint64_t key_hash) {
    if (!table) return false;
    
    size_t bucket_idx = key_hash % table->bucket_count;
    HashEntry* entry = table->buckets[bucket_idx];
    HashEntry* prev = NULL;
    
    while (entry) {
        if (entry->key_hash == key_hash) {
            if (prev) {
                prev->next = entry->next;
            } else {
                table->buckets[bucket_idx] = entry->next;
            }
            
            if (entry->cached_values) {
                free(entry->cached_values);
            }
            free(entry);
            
            table->entry_count--;
            table->load_factor = (double)table->entry_count / table->bucket_count;
            
            return true;
        }
        prev = entry;
        entry = entry->next;
    }
    
    return false;
}

// ============================================================================
// Cascading Hash Tables Implementation
// ============================================================================

CascadingHashTables* cascading_hash_create(size_t level_count, 
                                           size_t* level_sizes, 
                                           double* thresholds) {
    if (!level_sizes || !thresholds || level_count == 0) return NULL;
    
    CascadingHashTables* cascading = (CascadingHashTables*)malloc(sizeof(CascadingHashTables));
    if (!cascading) return NULL;
    
    cascading->levels = (HashTable**)malloc(level_count * sizeof(HashTable*));
    if (!cascading->levels) {
        free(cascading);
        return NULL;
    }
    
    cascading->level_sizes = (size_t*)malloc(level_count * sizeof(size_t));
    cascading->similarity_thresholds = (double*)malloc(level_count * sizeof(double));
    
    if (!cascading->level_sizes || !cascading->similarity_thresholds) {
        free(cascading->levels);
        free(cascading);
        return NULL;
    }
    
    cascading->level_count = level_count;
    
    for (size_t i = 0; i < level_count; i++) {
        cascading->levels[i] = hash_table_create(level_sizes[i]);
        if (!cascading->levels[i]) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                hash_table_free(cascading->levels[j]);
            }
            free(cascading->levels);
            free(cascading->level_sizes);
            free(cascading->similarity_thresholds);
            free(cascading);
            return NULL;
        }
        cascading->level_sizes[i] = level_sizes[i];
        cascading->similarity_thresholds[i] = thresholds[i];
    }
    
    return cascading;
}

void cascading_hash_free(CascadingHashTables* cascading) {
    if (!cascading) return;
    
    for (size_t i = 0; i < cascading->level_count; i++) {
        hash_table_free(cascading->levels[i]);
    }
    
    free(cascading->levels);
    free(cascading->level_sizes);
    free(cascading->similarity_thresholds);
    free(cascading);
}

int cascading_hash_insert(CascadingHashTables* cascading, 
                         uint64_t key_hash, 
                         GeneticProgram* solution, 
                         double similarity) {
    if (!cascading || !solution) return -1;
    
    // Find appropriate level based on similarity
    for (int i = (int)cascading->level_count - 1; i >= 0; i--) {
        if (similarity >= cascading->similarity_thresholds[i]) {
            if (hash_table_insert(cascading->levels[i], key_hash, solution)) {
                return i;
            }
        }
    }
    
    // Insert at lowest level (most permissive)
    if (hash_table_insert(cascading->levels[0], key_hash, solution)) {
        return 0;
    }
    
    return -1;
}

GeneticProgram* cascading_hash_lookup(CascadingHashTables* cascading, 
                                      uint64_t key_hash, 
                                      double min_similarity) {
    if (!cascading) return NULL;
    
    // Search from highest to lowest level
    for (int i = (int)cascading->level_count - 1; i >= 0; i--) {
        if (cascading->similarity_thresholds[i] >= min_similarity) {
            GeneticProgram* solution = hash_table_lookup(cascading->levels[i], key_hash);
            if (solution) {
                return solution;
            }
        }
    }
    
    return NULL;
}

void cascading_hash_stats(CascadingHashTables* cascading, 
                         size_t* total_entries, 
                         size_t* entries_per_level) {
    if (!cascading || !total_entries || !entries_per_level) return;
    
    *total_entries = 0;
    for (size_t i = 0; i < cascading->level_count; i++) {
        entries_per_level[i] = cascading->levels[i]->entry_count;
        *total_entries += entries_per_level[i];
    }
}

// ============================================================================
// Genetic Program Implementation
// ============================================================================

GPNode* gp_node_create(GPNodeType type, double value, int var_index) {
    GPNode* node = (GPNode*)malloc(sizeof(GPNode));
    if (!node) return NULL;
    
    node->type = type;
    node->value = value;
    node->var_index = var_index;
    node->left = NULL;
    node->right = NULL;
    
    return node;
}

void gp_node_free(GPNode* node) {
    if (!node) return;
    
    gp_node_free(node->left);
    gp_node_free(node->right);
    free(node);
}

static GPNode* gp_node_copy(GPNode* node) {
    if (!node) return NULL;
    
    GPNode* copy = gp_node_create(node->type, node->value, node->var_index);
    if (!copy) return NULL;
    
    copy->left = gp_node_copy(node->left);
    copy->right = gp_node_copy(node->right);
    
    return copy;
}

GeneticProgram* gp_copy(GeneticProgram* program) {
    if (!program) return NULL;
    
    GeneticProgram* copy = (GeneticProgram*)malloc(sizeof(GeneticProgram));
    if (!copy) return NULL;
    
    copy->root = gp_node_copy(program->root);
    copy->fitness = program->fitness;
    copy->complexity = program->complexity;
    copy->param_count = program->param_count;
    
    if (program->parameters && program->param_count > 0) {
        copy->parameters = (double*)malloc(program->param_count * sizeof(double));
        if (!copy->parameters) {
            gp_node_free(copy->root);
            free(copy);
            return NULL;
        }
        memcpy(copy->parameters, program->parameters, 
               program->param_count * sizeof(double));
    } else {
        copy->parameters = NULL;
    }
    
    return copy;
}

void gp_free(GeneticProgram* program) {
    if (!program) return;
    
    gp_node_free(program->root);
    if (program->parameters) {
        free(program->parameters);
    }
    free(program);
}

static double gp_node_evaluate(GPNode* node, double* vars, size_t var_count) {
    if (!node) return 0.0;
    
    switch (node->type) {
        case GP_NODE_CONSTANT:
            return node->value;
            
        case GP_NODE_VARIABLE:
            if (node->var_index >= 0 && (size_t)node->var_index < var_count) {
                return vars[node->var_index];
            }
            return 0.0;
            
        case GP_NODE_ADD: {
            double left = gp_node_evaluate(node->left, vars, var_count);
            double right = gp_node_evaluate(node->right, vars, var_count);
            return left + right;
        }
        
        case GP_NODE_SUB: {
            double left = gp_node_evaluate(node->left, vars, var_count);
            double right = gp_node_evaluate(node->right, vars, var_count);
            return left - right;
        }
        
        case GP_NODE_MUL: {
            double left = gp_node_evaluate(node->left, vars, var_count);
            double right = gp_node_evaluate(node->right, vars, var_count);
            return left * right;
        }
        
        case GP_NODE_DIV: {
            double left = gp_node_evaluate(node->left, vars, var_count);
            double right = gp_node_evaluate(node->right, vars, var_count);
            return (fabs(right) > 1e-10) ? (left / right) : 0.0;
        }
        
        case GP_NODE_POW: {
            double left = gp_node_evaluate(node->left, vars, var_count);
            double right = gp_node_evaluate(node->right, vars, var_count);
            return pow(left, right);
        }
        
        case GP_NODE_SIN: {
            double arg = gp_node_evaluate(node->left, vars, var_count);
            return sin(arg);
        }
        
        case GP_NODE_COS: {
            double arg = gp_node_evaluate(node->left, vars, var_count);
            return cos(arg);
        }
        
        case GP_NODE_EXP: {
            double arg = gp_node_evaluate(node->left, vars, var_count);
            return exp(arg);
        }
        
        case GP_NODE_LOG: {
            double arg = gp_node_evaluate(node->left, vars, var_count);
            return (arg > 0.0) ? log(arg) : 0.0;
        }
        
        case GP_NODE_SQRT: {
            double arg = gp_node_evaluate(node->left, vars, var_count);
            return (arg >= 0.0) ? sqrt(arg) : 0.0;
        }
        
        default:
            return 0.0;
    }
}

double gp_evaluate(GeneticProgram* program, double* vars, size_t var_count) {
    if (!program || !program->root) return 0.0;
    return gp_node_evaluate(program->root, vars, var_count);
}

static size_t gp_node_complexity(GPNode* node) {
    if (!node) return 0;
    
    size_t count = 1; // Count this node
    count += gp_node_complexity(node->left);
    count += gp_node_complexity(node->right);
    
    return count;
}

size_t gp_complexity(GeneticProgram* program) {
    if (!program || !program->root) return 0;
    return gp_node_complexity(program->root);
}

static double random_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static GPNode* gp_random_node(size_t depth, size_t max_depth, size_t var_count) {
    if (depth >= max_depth) {
        // Terminal node: constant or variable
        if (rand() % 2 == 0) {
            return gp_node_create(GP_NODE_CONSTANT, random_double(-10.0, 10.0), -1);
        } else {
            int var_idx = (int)(rand() % var_count);
            return gp_node_create(GP_NODE_VARIABLE, 0.0, var_idx);
        }
    }
    
    // Function node
    GPNodeType types[] = {
        GP_NODE_ADD, GP_NODE_SUB, GP_NODE_MUL, GP_NODE_DIV,
        GP_NODE_POW, GP_NODE_SIN, GP_NODE_COS, GP_NODE_EXP,
        GP_NODE_LOG, GP_NODE_SQRT
    };
    
    GPNodeType type = types[rand() % 10];
    GPNode* node = gp_node_create(type, 0.0, -1);
    
    if (type == GP_NODE_SIN || type == GP_NODE_COS || 
        type == GP_NODE_EXP || type == GP_NODE_LOG || type == GP_NODE_SQRT) {
        // Unary operators
        node->left = gp_random_node(depth + 1, max_depth, var_count);
    } else {
        // Binary operators
        node->left = gp_random_node(depth + 1, max_depth, var_count);
        node->right = gp_random_node(depth + 1, max_depth, var_count);
    }
    
    return node;
}

GeneticProgram* gp_random(size_t max_depth, size_t var_count) {
    GeneticProgram* program = (GeneticProgram*)malloc(sizeof(GeneticProgram));
    if (!program) return NULL;
    
    program->root = gp_random_node(0, max_depth, var_count);
    program->fitness = 0.0;
    program->parameters = NULL;
    program->param_count = 0;
    program->complexity = gp_complexity(program);
    
    return program;
}

// ============================================================================
// Genetic Solver Implementation
// ============================================================================

GeneticSolver* genetic_solver_create(ProblemDef* problem, 
                                     GAParams* ga_params,
                                     FitnessFunction fitness_func,
                                     void* user_data) {
    if (!problem || !ga_params || !fitness_func) return NULL;
    
    GeneticSolver* solver = (GeneticSolver*)malloc(sizeof(GeneticSolver));
    if (!solver) return NULL;
    
    memcpy(&solver->problem, problem, sizeof(ProblemDef));
    memcpy(&solver->ga_params, ga_params, sizeof(GAParams));
    
    solver->population_size = ga_params->population_size;
    solver->population = (GeneticProgram**)calloc(solver->population_size, 
                                                  sizeof(GeneticProgram*));
    if (!solver->population) {
        free(solver);
        return NULL;
    }
    
    solver->hash_tables = NULL;
    solver->fitness_func = fitness_func;
    solver->fitness_user_data = user_data;
    
    solver->generation = 0;
    solver->best_fitness = -INFINITY;
    solver->best_solution = NULL;
    solver->hash_hits = 0;
    solver->hash_misses = 0;
    solver->evaluations = 0;
    
    solver->offline_mode = false;
    solver->cache_file = NULL;
    
    return solver;
}

void genetic_solver_free(GeneticSolver* solver) {
    if (!solver) return;
    
    // Free population
    if (solver->population) {
        for (size_t i = 0; i < solver->population_size; i++) {
            if (solver->population[i]) {
                gp_free(solver->population[i]);
            }
        }
        free(solver->population);
    }
    
    // Free best solution
    if (solver->best_solution) {
        gp_free(solver->best_solution);
    }
    
    // Free hash tables
    if (solver->hash_tables) {
        cascading_hash_free(solver->hash_tables);
    }
    
    if (solver->cache_file) {
        free(solver->cache_file);
    }
    
    free(solver);
}

bool genetic_solver_init_population(GeneticSolver* solver) {
    if (!solver) return false;
    
    size_t var_count = (solver->problem.problem_type == 0) ? 
                       (1 + solver->problem.dimension) : // ODE: t + state vars
                       (1 + solver->problem.spatial_dim + solver->problem.dimension); // PDE
    
    for (size_t i = 0; i < solver->population_size; i++) {
        solver->population[i] = gp_random(5, var_count); // Max depth 5
        if (!solver->population[i]) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                gp_free(solver->population[j]);
            }
            return false;
        }
    }
    
    return true;
}

// Placeholder implementations for mutation, crossover, and evolution
// These would be fully implemented in a complete version

GeneticProgram* gp_mutate(GeneticProgram* program, double mutation_rate) {
    // Simplified: return a copy for now
    // Full implementation would mutate nodes
    return gp_copy(program);
}

void gp_crossover(GeneticProgram* parent1, 
                  GeneticProgram* parent2, 
                  GeneticProgram** child1, 
                  GeneticProgram** child2) {
    // Simplified: return copies for now
    // Full implementation would perform subtree crossover
    *child1 = gp_copy(parent1);
    *child2 = gp_copy(parent2);
}

bool genetic_solver_evolve_generation(GeneticSolver* solver) {
    // Placeholder: would implement full GA evolution
    return false;
}

GeneticProgram* genetic_solver_solve(GeneticSolver* solver) {
    if (!solver) return NULL;
    
    if (!genetic_solver_init_population(solver)) {
        return NULL;
    }
    
    // Placeholder: would run full GA
    return solver->best_solution;
}

bool genetic_solver_set_offline_mode(GeneticSolver* solver, const char* cache_file) {
    if (!solver) return false;
    
    solver->offline_mode = true;
    if (solver->cache_file) {
        free(solver->cache_file);
    }
    
    if (cache_file) {
        solver->cache_file = strdup(cache_file);
    }
    
    return true;
}

bool genetic_solver_load_cache(GeneticSolver* solver, const char* cache_file) {
    // Placeholder: would load from file
    (void)solver;
    (void)cache_file;
    return false;
}

bool genetic_solver_save_cache(GeneticSolver* solver, const char* cache_file) {
    // Placeholder: would save to file
    (void)solver;
    (void)cache_file;
    return false;
}

void genetic_solver_get_stats(GeneticSolver* solver, 
                             size_t* generation,
                             double* best_fitness,
                             size_t* hash_hits,
                             size_t* hash_misses,
                             size_t* evaluations) {
    if (!solver) return;
    
    if (generation) *generation = solver->generation;
    if (best_fitness) *best_fitness = solver->best_fitness;
    if (hash_hits) *hash_hits = solver->hash_hits;
    if (hash_misses) *hash_misses = solver->hash_misses;
    if (evaluations) *evaluations = solver->evaluations;
}

bool genetic_solver_set_hash_tables(GeneticSolver* solver,
                                   size_t level_count,
                                   size_t* level_sizes,
                                   double* thresholds) {
    if (!solver) return false;
    
    if (solver->hash_tables) {
        cascading_hash_free(solver->hash_tables);
    }
    
    solver->hash_tables = cascading_hash_create(level_count, level_sizes, thresholds);
    return solver->hash_tables != NULL;
}
