namespace Model

module ModelActions =
    open System
    open System.IO
    open Microsoft.ML
    open Microsoft.ML.Transforms.Text

    let createSets<'T>(ctx : MLContext, dataPath : string) =
        let dataView = ctx.Data.LoadFromTextFile<'T>(dataPath, hasHeader = true)

        let trainTestSplit = ctx.Data.TrainTestSplit(dataView, testFraction=0.2)

        (trainTestSplit.TrainSet, trainTestSplit.TestSet)

    let createDataProcessPipeline<'T>(ctx : MLContext, trainSet : IDataView, peekData : bool) =
        let pipeline = ctx.Transforms.Text.FeaturizeText("Features", "Text")

        if peekData then
            // (OPTIONAL) Peek data (such as 2 records) in training DataView after applying the ProcessPipeline's transformations into "Features"
            Common.ConsoleHelper.peekDataViewInConsole ctx trainSet pipeline 2 |> ignore

            //Peak the transformed features column
            Common.ConsoleHelper.peekVectorColumnDataInConsole ctx "Features" trainSet pipeline 1 |> ignore

        pipeline

    let createTrainingPipeline(ctx : MLContext, pipeline : TextFeaturizingEstimator)=
        let trainer = ctx.BinaryClassification.Trainers.FastTree(labelColumnName = "Label", featureColumnName = "Features")
        let trainingPipeline = pipeline.Append(trainer)
        trainingPipeline

    let train (trainSet : IDataView, trainingPipeline : Data.EstimatorChain<_>) =
        printfn "=============== Training the model ==============="
        let trainedModel = trainingPipeline.Fit(trainSet)
        trainedModel

    let evaluate (ctx : MLContext , trainedModel : Data.TransformerChain<_>, testSet : IDataView)=
        printfn "=== Evaluating Model's accuracy with Test data ==="
        let predictions = trainedModel.Transform testSet
        let metrics = ctx.BinaryClassification.Evaluate(predictions, "Label", "Score")
        metrics

    let persist (ctx : MLContext, modelPath : string, trainedModel : ITransformer, trainSet : IDataView)=
        use fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write)
        ctx.Model.Save(trainedModel, trainSet.Schema, fs)

        printfn "The model is saved to %s" modelPath

    let testPrediction<'TStatement,'TPrediction when 'TStatement: not struct and 'TPrediction: (new: unit -> 'TPrediction) and 'TPrediction: not struct> (ctx : MLContext, sampleStatement : 'TStatement, modelPath : string) =

        use stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read)
        let trainedModel, inputSchema = ctx.Model.Load(stream)

        // Create prediction engine related to the loaded trained model
        let predEngine= ctx.Model.CreatePredictionEngine<'TStatement, 'TPrediction>(trainedModel)

        //Score
        let resultprediction = predEngine.Predict(sampleStatement)
        resultprediction