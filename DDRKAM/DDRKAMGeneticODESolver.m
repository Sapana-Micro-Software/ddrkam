//
//  DDRKAMGeneticODESolver.m
//  DDRKAM
//
//  Genetic Programming Solver for ODEs and PDEs
//  Uses cascading hash tables for offline caching
//

#import "DDRKAMGeneticODESolver.h"
#import "genetic_ode_solver.h"
#import <math.h>

// ============================================================================
// Problem Definition
// ============================================================================

@implementation DDRKAMProblemDefinition

- (instancetype)initWithType:(DDRKAMProblemType)type
                   dimension:(NSInteger)dim
            spatialDimension:(NSInteger)spatialDim
         initialConditions:(NSArray<NSNumber*>*)initialConditions
              domainBounds:(NSArray<NSNumber*>*)domainBounds
                tolerance:(double)tolerance
                  maxSteps:(NSUInteger)maxSteps {
    self = [super init];
    if (self) {
        _problemType = type;
        _dimension = dim;
        _spatialDimension = spatialDim;
        _initialConditions = [initialConditions copy];
        _domainBounds = [domainBounds copy];
        _tolerance = tolerance;
        _maxSteps = maxSteps;
    }
    return self;
}

@end

// ============================================================================
// GA Parameters
// ============================================================================

@implementation DDRKAMGAParameters

+ (instancetype)defaultParameters {
    DDRKAMGAParameters* params = [[DDRKAMGAParameters alloc] init];
    params.populationSize = 100;
    params.maxGenerations = 1000;
    params.mutationRate = 0.1;
    params.crossoverRate = 0.7;
    params.selectionPressure = 2.0;
    params.tournamentSize = 5;
    params.elitismCount = 5;
    params.minFitness = 1e-6;
    params.useParallel = NO;
    return params;
}

@end

// ============================================================================
// Genetic Program
// ============================================================================

@implementation DDRKAMGeneticProgram {
    GeneticProgram* _cProgram;
}

- (instancetype)initWithCProgram:(GeneticProgram*)cProgram {
    self = [super init];
    if (self) {
        _cProgram = cProgram;
    }
    return self;
}

- (void)dealloc {
    if (_cProgram) {
        gp_free(_cProgram);
    }
}

- (double)fitness {
    return _cProgram ? _cProgram->fitness : 0.0;
}

- (NSArray<NSNumber*>*)parameters {
    if (!_cProgram || !_cProgram->parameters || _cProgram->param_count == 0) {
        return @[];
    }
    
    NSMutableArray* params = [NSMutableArray arrayWithCapacity:_cProgram->param_count];
    for (size_t i = 0; i < _cProgram->param_count; i++) {
        [params addObject:@(_cProgram->parameters[i])];
    }
    return [params copy];
}

- (NSUInteger)complexity {
    return _cProgram ? (NSUInteger)_cProgram->complexity : 0;
}

- (double)evaluateWithVariables:(NSArray<NSNumber*>*)variables {
    if (!_cProgram || !variables) return 0.0;
    
    size_t varCount = variables.count;
    double* vars = (double*)malloc(varCount * sizeof(double));
    if (!vars) return 0.0;
    
    for (size_t i = 0; i < varCount; i++) {
        vars[i] = [variables[i] doubleValue];
    }
    
    double result = gp_evaluate(_cProgram, vars, varCount);
    free(vars);
    
    return result;
}

- (DDRKAMGeneticProgram*)copy {
    if (!_cProgram) return nil;
    
    GeneticProgram* copied = gp_copy(_cProgram);
    if (!copied) return nil;
    
    return [[DDRKAMGeneticProgram alloc] initWithCProgram:copied];
}

@end

// ============================================================================
// Cascading Hash Tables
// ============================================================================

@implementation DDRKAMCascadingHashTables {
    CascadingHashTables* _cCascading;
    NSArray<NSNumber*>* _entriesPerLevel;
}

- (instancetype)initWithLevelCount:(NSUInteger)levelCount
                         levelSizes:(NSArray<NSNumber*>*)levelSizes
                 similarityThresholds:(NSArray<NSNumber*>*)thresholds {
    self = [super init];
    if (self) {
        if (levelCount == 0 || levelSizes.count != levelCount || 
            thresholds.count != levelCount) {
            return nil;
        }
        
        size_t* sizes = (size_t*)malloc(levelCount * sizeof(size_t));
        double* thresh = (double*)malloc(levelCount * sizeof(double));
        
        if (!sizes || !thresh) {
            free(sizes);
            free(thresh);
            return nil;
        }
        
        for (NSUInteger i = 0; i < levelCount; i++) {
            sizes[i] = (size_t)[levelSizes[i] unsignedIntegerValue];
            thresh[i] = [thresholds[i] doubleValue];
        }
        
        _cCascading = cascading_hash_create(levelCount, sizes, thresh);
        
        free(sizes);
        free(thresh);
        
        if (!_cCascading) {
            return nil;
        }
        
        _levelCount = levelCount;
        [self updateStats];
    }
    return self;
}

- (void)dealloc {
    if (_cCascading) {
        cascading_hash_free(_cCascading);
    }
}

- (void)updateStats {
    if (!_cCascading) return;
    
    size_t* entriesPerLevel = (size_t*)malloc(_levelCount * sizeof(size_t));
    size_t totalEntries = 0;
    
    cascading_hash_stats(_cCascading, &totalEntries, entriesPerLevel);
    
    NSMutableArray* entries = [NSMutableArray arrayWithCapacity:_levelCount];
    for (NSUInteger i = 0; i < _levelCount; i++) {
        [entries addObject:@(entriesPerLevel[i])];
    }
    _entriesPerLevel = [entries copy];
    _totalEntries = (NSUInteger)totalEntries;
    
    free(entriesPerLevel);
}

- (NSInteger)insertSolution:(DDRKAMGeneticProgram*)solution
                  forKeyHash:(uint64_t)keyHash
                  similarity:(double)similarity {
    if (!_cCascading || !solution) return -1;
    
    // Get C program from Objective-C wrapper
    // Note: This requires exposing the internal C program pointer
    // For now, we'll need to store it or pass it differently
    // This is a simplified version
    
    return -1; // Placeholder
}

- (nullable DDRKAMGeneticProgram*)lookupForKeyHash:(uint64_t)keyHash
                                    minSimilarity:(double)minSimilarity {
    if (!_cCascading) return nil;
    
    GeneticProgram* cProgram = cascading_hash_lookup(_cCascading, keyHash, minSimilarity);
    if (!cProgram) return nil;
    
    return [[DDRKAMGeneticProgram alloc] initWithCProgram:cProgram];
}

@end

// ============================================================================
// Genetic Solver
// ============================================================================

@implementation DDRKAMGeneticODESolver {
    GeneticSolver* _cSolver;
    double (^_fitnessFunction)(DDRKAMGeneticProgram*, DDRKAMProblemDefinition*);
}

- (instancetype)initWithProblem:(DDRKAMProblemDefinition*)problem
                     parameters:(DDRKAMGAParameters*)parameters
                 fitnessFunction:(double (^)(DDRKAMGeneticProgram* program, 
                                            DDRKAMProblemDefinition* problem))fitnessFunction {
    self = [super init];
    if (self) {
        _problem = problem;
        _parameters = parameters;
        _fitnessFunction = [fitnessFunction copy];
        
        // Convert Objective-C problem to C structure
        ProblemDef cProblem;
        cProblem.problem_type = (int)problem.problemType;
        cProblem.dimension = (int)problem.dimension;
        cProblem.spatial_dim = (int)problem.spatialDimension;
        cProblem.tolerance = problem.tolerance;
        cProblem.max_steps = (size_t)problem.maxSteps;
        
        // Convert initial conditions
        if (problem.initialConditions.count > 0) {
            cProblem.initial_conditions = (double*)malloc(problem.initialConditions.count * sizeof(double));
            for (NSUInteger i = 0; i < problem.initialConditions.count; i++) {
                cProblem.initial_conditions[i] = [problem.initialConditions[i] doubleValue];
            }
        } else {
            cProblem.initial_conditions = NULL;
        }
        
        // Convert domain bounds
        if (problem.domainBounds.count > 0) {
            cProblem.domain_bounds = (double*)malloc(problem.domainBounds.count * sizeof(double));
            for (NSUInteger i = 0; i < problem.domainBounds.count; i++) {
                cProblem.domain_bounds[i] = [problem.domainBounds[i] doubleValue];
            }
        } else {
            cProblem.domain_bounds = NULL;
        }
        
        // Convert GA parameters
        GAParams cParams;
        cParams.population_size = (size_t)parameters.populationSize;
        cParams.max_generations = (size_t)parameters.maxGenerations;
        cParams.mutation_rate = parameters.mutationRate;
        cParams.crossover_rate = parameters.crossoverRate;
        cParams.selection_pressure = parameters.selectionPressure;
        cParams.tournament_size = (size_t)parameters.tournamentSize;
        cParams.elitism_count = (size_t)parameters.elitismCount;
        cParams.min_fitness = parameters.minFitness;
        cParams.use_parallel = parameters.useParallel;
        
        // Create fitness function wrapper
        FitnessFunction cFitnessFunc = [](GeneticProgram* program, ProblemDef* problem, void* user_data) -> double {
            DDRKAMGeneticODESolver* solver = (__bridge DDRKAMGeneticODESolver*)user_data;
            DDRKAMGeneticProgram* ocProgram = [[DDRKAMGeneticProgram alloc] initWithCProgram:program];
            return solver->_fitnessFunction(ocProgram, solver->_problem);
        };
        
        _cSolver = genetic_solver_create(&cProblem, &cParams, cFitnessFunc, (__bridge void*)self);
        
        // Cleanup temporary arrays
        if (cProblem.initial_conditions) free(cProblem.initial_conditions);
        if (cProblem.domain_bounds) free(cProblem.domain_bounds);
        
        if (!_cSolver) {
            return nil;
        }
    }
    return self;
}

- (void)dealloc {
    if (_cSolver) {
        genetic_solver_free(_cSolver);
    }
}

- (NSUInteger)generation {
    return _cSolver ? (NSUInteger)_cSolver->generation : 0;
}

- (double)bestFitness {
    return _cSolver ? _cSolver->best_fitness : -INFINITY;
}

- (nullable DDRKAMGeneticProgram*)bestSolution {
    if (!_cSolver || !_cSolver->best_solution) return nil;
    return [[DDRKAMGeneticProgram alloc] initWithCProgram:_cSolver->best_solution];
}

- (NSUInteger)hashHits {
    return _cSolver ? (NSUInteger)_cSolver->hash_hits : 0;
}

- (NSUInteger)hashMisses {
    return _cSolver ? (NSUInteger)_cSolver->hash_misses : 0;
}

- (NSUInteger)evaluations {
    return _cSolver ? (NSUInteger)_cSolver->evaluations : 0;
}

- (BOOL)initializePopulation {
    return _cSolver ? genetic_solver_init_population(_cSolver) : NO;
}

- (BOOL)evolveGeneration {
    return _cSolver ? genetic_solver_evolve_generation(_cSolver) : NO;
}

- (nullable DDRKAMGeneticProgram*)solve {
    if (!_cSolver) return nil;
    
    GeneticProgram* cSolution = genetic_solver_solve(_cSolver);
    if (!cSolution) return nil;
    
    return [[DDRKAMGeneticProgram alloc] initWithCProgram:cSolution];
}

- (BOOL)setHashTablesWithLevelCount:(NSUInteger)levelCount
                          levelSizes:(NSArray<NSNumber*>*)levelSizes
                 similarityThresholds:(NSArray<NSNumber*>*)thresholds {
    if (!_cSolver || levelSizes.count != levelCount || thresholds.count != levelCount) {
        return NO;
    }
    
    size_t* sizes = (size_t*)malloc(levelCount * sizeof(size_t));
    double* thresh = (double*)malloc(levelCount * sizeof(double));
    
    if (!sizes || !thresh) {
        free(sizes);
        free(thresh);
        return NO;
    }
    
    for (NSUInteger i = 0; i < levelCount; i++) {
        sizes[i] = (size_t)[levelSizes[i] unsignedIntegerValue];
        thresh[i] = [thresholds[i] doubleValue];
    }
    
    BOOL result = genetic_solver_set_hash_tables(_cSolver, levelCount, sizes, thresh);
    
    free(sizes);
    free(thresh);
    
    if (result) {
        // Update Objective-C wrapper
        if (_cSolver->hash_tables) {
            _hashTables = [[DDRKAMCascadingHashTables alloc] 
                          initWithLevelCount:levelCount
                                   levelSizes:levelSizes
                        similarityThresholds:thresholds];
        }
    }
    
    return result;
}

- (BOOL)loadCacheFromFile:(NSString*)filePath {
    if (!_cSolver || !filePath) return NO;
    return genetic_solver_load_cache(_cSolver, [filePath UTF8String]);
}

- (BOOL)saveCacheToFile:(NSString*)filePath {
    if (!_cSolver || !filePath) return NO;
    return genetic_solver_save_cache(_cSolver, [filePath UTF8String]);
}

@end
