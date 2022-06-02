open System
open System.IO
open Microsoft.ML
open DataStructures.Model
open Model.ModelActions

let dataPath = Path.Combine(__SOURCE_DIRECTORY__, "Data\\wikiDetoxAnnotated40kRows.tsv.txt")
let modelPath = Path.Combine(__SOURCE_DIRECTORY__, "MLModels\\SentimentModel.zip")

let runML(mlContext: MLContext)=
    // STEP 1: Common data loading configuration
    let (trainSet, testSet) = createSets<SentimentIssue>(mlContext, dataPath)

    // STEP 2: Common data process configuration with pipeline data transformations
    let dataProcessPipeline = createDataProcessPipeline<SentimentIssue>(mlContext, trainSet, true)

    // STEP 3: Set the training algorithm, then create and config the modelBuilder
    let trainingPipeline = createTrainingPipeline(mlContext, dataProcessPipeline)

    // STEP 4: Train the model fitting to the DataSet
    let trainedModel = train(trainSet, trainingPipeline)

    // STEP 5: Evaluate the model and show accuracy stats
    let metrics = evaluate(mlContext, trainedModel, testSet)

    // STEP 6: Save/persist the trained model to a .ZIP file
    persist(mlContext, modelPath, trainedModel, trainSet)

    Common.ConsoleHelper.consoleWriteHeader "=============== End of training process ==============="

//Create MLContext to be shared across the model creation workflow objects
//Set a random seed for repeatable/deterministic results across multiple trainings.
let mlContext = MLContext(seed = Nullable 1)

printf "Create, Train, Evaluate and Save a model ? [y/N]"
let trainM = Console.ReadLine()

match trainM.ToLower() with
| "y" -> runML(mlContext)
| _ -> printfn "N"

// Make a single test prediction loding the model from .ZIP file
printf "Statement: "
let statementText = Console.ReadLine()

let sampleStatement = { Label = true; Text = statementText }
let resultprediction = testPrediction(mlContext, sampleStatement, modelPath)

let sentiment = Convert.ToBoolean(resultprediction.Prediction)

printfn "=============== Single Prediction  ==============="
printfn "Text: %s | Prediction: %s sentiment | Probability: %f" sampleStatement.Text (if sentiment then "Negative" else "Positive") resultprediction.Probability
printfn "=================================================="

Common.ConsoleHelper.consoleWriteHeader "=============== End of process, hit any key to finish ==============="
Common.ConsoleHelper.consolePressAnyKey()