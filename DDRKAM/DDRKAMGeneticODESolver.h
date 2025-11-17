//
//  DDRKAMGeneticODESolver.h
//  DDRKAM
//
//  Genetic Programming Solver for ODEs and PDEs
//  Uses cascading hash tables for offline caching
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Problem types
typedef NS_ENUM(NSInteger, DDRKAMProblemType) {
    DDRKAMProblemTypeODE = 0,
    DDRKAMProblemTypePDE = 1
};

// Genetic Program Node Types
typedef NS_ENUM(NSInteger, DDRKAMGPNodeType) {
    DDRKAMGPNodeTypeConstant,
    DDRKAMGPNodeTypeVariable,
    DDRKAMGPNodeTypeAdd,
    DDRKAMGPNodeTypeSub,
    DDRKAMGPNodeTypeMul,
    DDRKAMGPNodeTypeDiv,
    DDRKAMGPNodeTypePow,
    DDRKAMGPNodeTypeSin,
    DDRKAMGPNodeTypeCos,
    DDRKAMGPNodeTypeExp,
    DDRKAMGPNodeTypeLog,
    DDRKAMGPNodeTypeSqrt
};

@class DDRKAMGeneticProgram;
@class DDRKAMProblemDefinition;
@class DDRKAMGAParameters;

// Problem Definition
@interface DDRKAMProblemDefinition : NSObject

@property (nonatomic, assign) DDRKAMProblemType problemType;
@property (nonatomic, assign) NSInteger dimension;
@property (nonatomic, assign) NSInteger spatialDimension;
@property (nonatomic, strong) NSArray<NSNumber*>* initialConditions;
@property (nonatomic, strong) NSArray<NSNumber*>* domainBounds;
@property (nonatomic, assign) double tolerance;
@property (nonatomic, assign) NSUInteger maxSteps;

- (instancetype)initWithType:(DDRKAMProblemType)type
                   dimension:(NSInteger)dim
            spatialDimension:(NSInteger)spatialDim
         initialConditions:(NSArray<NSNumber*>*)initialConditions
              domainBounds:(NSArray<NSNumber*>*)domainBounds
                tolerance:(double)tolerance
                  maxSteps:(NSUInteger)maxSteps;

@end

// Genetic Algorithm Parameters
@interface DDRKAMGAParameters : NSObject

@property (nonatomic, assign) NSUInteger populationSize;
@property (nonatomic, assign) NSUInteger maxGenerations;
@property (nonatomic, assign) double mutationRate;
@property (nonatomic, assign) double crossoverRate;
@property (nonatomic, assign) double selectionPressure;
@property (nonatomic, assign) NSUInteger tournamentSize;
@property (nonatomic, assign) NSUInteger elitismCount;
@property (nonatomic, assign) double minFitness;
@property (nonatomic, assign) BOOL useParallel;

+ (instancetype)defaultParameters;

@end

// Genetic Program
@interface DDRKAMGeneticProgram : NSObject

@property (nonatomic, readonly) double fitness;
@property (nonatomic, readonly) NSArray<NSNumber*>* parameters;
@property (nonatomic, readonly) NSUInteger complexity;

- (double)evaluateWithVariables:(NSArray<NSNumber*>*)variables;
- (DDRKAMGeneticProgram*)copy;

@end

// Cascading Hash Tables
@interface DDRKAMCascadingHashTables : NSObject

@property (nonatomic, readonly) NSUInteger levelCount;
@property (nonatomic, readonly) NSArray<NSNumber*>* entriesPerLevel;
@property (nonatomic, readonly) NSUInteger totalEntries;

- (instancetype)initWithLevelCount:(NSUInteger)levelCount
                         levelSizes:(NSArray<NSNumber*>*)levelSizes
                    similarityThresholds:(NSArray<NSNumber*>*)thresholds;

- (NSInteger)insertSolution:(DDRKAMGeneticProgram*)solution
                  forKeyHash:(uint64_t)keyHash
                  similarity:(double)similarity;

- (nullable DDRKAMGeneticProgram*)lookupForKeyHash:(uint64_t)keyHash
                                    minSimilarity:(double)minSimilarity;

@end

// Genetic Solver
@interface DDRKAMGeneticODESolver : NSObject

@property (nonatomic, strong, readonly) DDRKAMProblemDefinition* problem;
@property (nonatomic, strong, readonly) DDRKAMGAParameters* parameters;
@property (nonatomic, strong, readonly, nullable) DDRKAMCascadingHashTables* hashTables;
@property (nonatomic, assign, readonly) NSUInteger generation;
@property (nonatomic, assign, readonly) double bestFitness;
@property (nonatomic, strong, readonly, nullable) DDRKAMGeneticProgram* bestSolution;
@property (nonatomic, assign, readonly) NSUInteger hashHits;
@property (nonatomic, assign, readonly) NSUInteger hashMisses;
@property (nonatomic, assign, readonly) NSUInteger evaluations;
@property (nonatomic, assign) BOOL offlineMode;
@property (nonatomic, strong, nullable) NSString* cacheFilePath;

- (instancetype)initWithProblem:(DDRKAMProblemDefinition*)problem
                     parameters:(DDRKAMGAParameters*)parameters
                 fitnessFunction:(double (^)(DDRKAMGeneticProgram* program, 
                                            DDRKAMProblemDefinition* problem))fitnessFunction;

- (BOOL)initializePopulation;

- (BOOL)evolveGeneration;

- (nullable DDRKAMGeneticProgram*)solve;

- (BOOL)setHashTablesWithLevelCount:(NSUInteger)levelCount
                          levelSizes:(NSArray<NSNumber*>*)levelSizes
                 similarityThresholds:(NSArray<NSNumber*>*)thresholds;

- (BOOL)loadCacheFromFile:(NSString*)filePath;

- (BOOL)saveCacheToFile:(NSString*)filePath;

@end

NS_ASSUME_NONNULL_END
