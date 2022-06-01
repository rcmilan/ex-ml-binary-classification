open System
open System.IO
open Microsoft.ML
open DataStructures.Model
open Model.ModelActions

let baseDirectory = __SOURCE_DIRECTORY__
let dataPath = Path.Combine(baseDirectory, "Data\\wikiDetoxAnnotated40kRows.tsv.txt")

let modelPath = Path.Combine(baseDirectory, "MLModels\\SentimentModel.zip")



//Create MLContext to be shared across the model creation workflow objects
//Set a random seed for repeatable/deterministic results across multiple trainings.
let mlContext = MLContext(seed = Nullable 1)

// Create, Train, Evaluate and Save a model
// STEP 1: Common data loading configuration
let (trainSet, testSet) = createSets<SentimentIssue>(mlContext, dataPath)

// STEP 2: Common data process configuration with pipeline data transformations
let dataProcessPipeline = createDataProcessPipeline<SentimentIssue>(mlContext, trainSet, false)

// STEP 3: Set the training algorithm, then create and config the modelBuilder
let trainingPipeline = createTrainingPipeline(mlContext, dataProcessPipeline)

// STEP 4: Train the model fitting to the DataSet
let trainedModel = train(trainSet, trainingPipeline)

// STEP 5: Evaluate the model and show accuracy stats
let metrics = evaluate(mlContext, trainedModel, testSet)

// STEP 6: Save/persist the trained model to a .ZIP file
persist(mlContext, modelPath, trainedModel, trainSet)

Common.ConsoleHelper.consoleWriteHeader "=============== End of training processh ==============="

// Make a single test prediction loding the model from .ZIP file
let sampleStatement = { Label = false; Text = "This is a very rude movie" }
let resultprediction = testPrediction(mlContext, sampleStatement, modelPath)

printfn "=============== Single Prediction  ==============="
printfn
    "Text: %s | Prediction: %s sentiment | Probability: %f"
    sampleStatement.Text
    (if Convert.ToBoolean(resultprediction.Prediction) then "Negative" else "Positive")
    resultprediction.Probability
printfn "=================================================="

Common.ConsoleHelper.consoleWriteHeader "=============== End of process, hit any key to finish ==============="
Common.ConsoleHelper.consolePressAnyKey()