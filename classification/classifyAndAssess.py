import ee
import math
import configparser as cp
import io
import ast

from areaOfInterestMask import semiarid as sa
from featuresForModeling import generateFeatures as gf

def trainAndPredict(featuresOptions = None, classifierOptions = None, roiForClassification = None, resultNewFolderName = None, returnExisting = False, startFreshExport = False):
    def selectClassifier(opts):
        defaultMaxNodes = 300
        # Default RandomForest
        if opts.get("classifier") == "RandomForest":
            c = ee.Classifier.smileRandomForest(** {"numberOfTrees": opts.get("numTrees"), "maxNodes": defaultMaxNodes})
        elif opts.get("classifier") == "GradientBoostedTrees":
            c = ee.Classifier.smileGradientTreeBoost(** {"numberOfTrees": opts.get("numTrees"), "maxNodes": defaultMaxNodes})
        else:
            c = ee.Classifier.smileRandomForest(** {"numberOfTrees": opts.get("numTrees"), "maxNodes": defaultMaxNodes})
        return c

    def findtop1LabelNum(f):
        probs = f.getArray("classification")

        top1LabelVal = probs.argmax().getNumber(0).add(1)
        top1Label = labelsSortedByValues.getString(top1LabelVal.subtract(1)).slice(len("prob_"))

        probsList = probs.project([0]).toList()
        probsDict = ee.Dictionary.fromLists(labelsSortedByValues, probsList)

        inputLabelNum = f.getNumber(propNameWithNumLabelsToPredictOn)
        # Since the string version of the label is not in the feature f (not in the "select"ed table to predict on itself),
        # it is not directly accessible like the Num version above.
        # Hence, picked up from the master table itself.
        inputLabel = ee.Feature(f.get("matchingPoint")).getString(propNameWithLabels)

        retFeat = ee.Feature(f.geometry(), probsDict.combine(ee.Dictionary({"top1LabelNum": top1LabelVal, "top1Label": top1Label, propNameWithLabels: inputLabel}))) \
            .copyProperties(** {'source': f, 'exclude': ["classification", "matchingPoint", "dist"]})
        return retFeat
        # return f.set(probsDict.combine(ee.Dictionary({"top1LabelNum": top1LabelVal, "top1Label": top1Label, "label": inputLabel}))) \

    def gatherPropsFromFeatCollIntoFeat(feat, faccum):
        return ee.Feature(faccum).set(feat.toDictionary())

    def calcClassifierAccuracies(labeledTable, scopeTag):
        predictionAssessment = labeledTable.errorMatrix(** {
            "actual": propNameWithNumLabelsToPredictOn,
            "predicted": "top1LabelNum",
            "order": labelvaluesInFilteredRun})
        accuraciesSummary = dict( \
            spanRegion = scopeTag, \
            errMatrixAccuracy = predictionAssessment.accuracy(), \
            errMatrixKappa = predictionAssessment.kappa(), \
            errMatrixConsumersAccuracy = predictionAssessment.consumersAccuracy().project([1]).toList().join(","), \
            errMatrixProducersAccuracy = predictionAssessment.producersAccuracy().project([0]).toList().join(","), \
            errMatrixNameAndOrderOfRowsAndCols = predictionAssessment.order().join(","), \
            classifier = classifierOptions.get("classifier"), \
            numClassifierTrees = classifierOptions.get("numTrees"))
        dummyPoint = ee.Geometry.Point([-65.05351562500002, 37.24376726222654]);
        accuraciesFeat = ee.Feature(dummyPoint, accuraciesSummary)

        return accuraciesFeat

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configCore = config["CORE"]
    assetFolder     = configCore.get('assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialRoiForClassificationDistrictName = configCore.get('trialSpanToGenerateFeatureRastersOverDistrict')
    trialRoiForClassification = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialRoiForClassificationDistrictName)).first())
    fullRoiForClassification  = ee.Feature(ee.FeatureCollection(configCore.get('fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)
    configAoi = config["AOI"]
    areaOfInterestBandname = configAoi.get("bandNameAOI")
    configFeaturesAssemble = config["FEATURES-ASSEMBLE"]
    wlONECategoriesInPtsList = ast.literal_eval(configFeaturesAssemble.get("wastelandAtlasCategoriesInExistingPointsTable"))
    wlONECategoriesColName = configFeaturesAssemble.get("wastelandAtlasLabelColumnName")
    lulcPalsarHarmnCategoriesInPtsList = ast.literal_eval(configFeaturesAssemble.get("lulcLabelsPalsarHarmonizedForONEMulticlass"))
    lulcPalsarHarmnCategoriesInPtsList_nonONEsSubset = ast.literal_eval(configFeaturesAssemble.get("lulcLabelsPalsarHarmonizedForONEMulticlass_nonONEsSubset"))
    lulcPalsarHarmnCategoriesColName = "lulcLabel"
    existingResultFolderName = config.get('CLASSIFICATION-TRAIN&PREDICT', 'existingResultFolderForPredictionsAndAccuracies')

    zoneNumSuff = 'Num'
    zoneOheSuff = 'Ohe_'
    if featuresOptions.get("zonationBasis") == "states":
        configClassifZones      = config["AOI-CLASSIFICATION-ZONES-STATES"]
        zonePref = configClassifZones.get("featureBandNamePrefix")
        zonesNumericFeaturename  = zonePref + zoneNumSuff
        zoneGroupsOfStatesLabels = list(ast.literal_eval(configClassifZones.get("groupsOfStatesLabels")))
        zoneOheFeatureNames    = [zonePref + zoneOheSuff + label for label in zoneGroupsOfStatesLabels]
    elif featuresOptions.get("zonationBasis") == "biomes":
        configClassifZones      = config["AOI-CLASSIFICATION-ZONES-BIOMES"]
        zonesNumericFeaturename = configClassifZones.get("featureBandNameZoneNumeric")
        zonesOheBandnamePrefix  = configClassifZones.get("featureBandNameOneHotEncodedZonePrefix")
    elif featuresOptions.get("zonationBasis") == "geologicalAge":
        configClassifZones      = config["AOI-CLASSIFICATION-ZONES-GEOLOGICAL-AGE"]
        zonesNumericFeaturename = configClassifZones.get("featureBandNameZoneNumeric")
        zonesOheBandnamePrefix  = configClassifZones.get("featureBandNameOneHotEncodedZonePrefix")
    elif featuresOptions.get("zonationBasis") == None:
        zonesNumericFeaturename = None
        zonesOheBandnamePrefix  = None

    labelsList = lulcPalsarHarmnCategoriesInPtsList
    propNameWithLabels = lulcPalsarHarmnCategoriesColName
    propNameWithNumLabelsToPredictOn = propNameWithLabels + "Num"
    if returnExisting == True:
        featureRasterPredictions = ee.Image(assetFolder + existingResultFolderName + '/prediction')
    else:
        areaOfInterest = sa.maskWithClassLabels(returnExisting = True)

        # Gather ALL feature rasters, as bands of a single image
        allFeaturesComposite = gf.assembleAllExistingFeatureRasters()
            # .updateMask(areaOfInterest.mask())

        # Bring in table with labeled points sampled on ALL feature rasters
        # and drop those outside the AOI
        pointsWithAllFeatures = gf.assembleFeatureBandsAndExport(returnExisting = True)

        # Decide what roi to run train/test/predict of classification on
        if roiForClassification == None:
            # reg = trialRoiForClassification
            reg = fullRoiForClassification
        else:
            reg = roiForClassification
        # Select the requisite bands/props specified by the user,
        # prepare the points by dropping nulls and randomizing
        featuresCompositeToRunWith = allFeaturesComposite.select(featuresOptions.get("names"))
            # .clip(reg.geometry())
            # .filterBounds(reg.geometry()) \
        tableWithFeaturesToRunWith = pointsWithAllFeatures.select(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOn]) \
            .filter(ee.Filter.notNull(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOn])) \
            .randomColumn()

        # Split points into train and test sets. Train on train set.
        trainPtsSplitFilter = ee.Filter.lte('random', classifierOptions.get("trainFraction"))
        trainingFraction = tableWithFeaturesToRunWith \
            .filter(trainPtsSplitFilter)
        testingFraction  = tableWithFeaturesToRunWith \
            .filter(trainPtsSplitFilter.Not())
        classifier = selectClassifier(classifierOptions)
        trainedModel = classifier.train(** { \
            'features': trainingFraction, \
            'classProperty': propNameWithNumLabelsToPredictOn, \
            'inputProperties': featuresOptions.get("names"),
        }).setOutputMode('MULTIPROBABILITY')

        # Predict over the full feature raster to produce prediction probabilities.
        featureRasterPredicted = ee.Image(featuresCompositeToRunWith.classify(trainedModel, "classProbabilities"))

        # Convert the array pixel image into multiband image, where band names correspond to the label names
        # First, sort NameList by ValList; NameList and ValList must be arranged such that
        # name-val correspondence is correct
        labelNameList = ee.List(labelsList)
        labelValList = [i for i in range(1, len(labelsList)+1)]

        allLabelnamesAndTheirNumLabels = ee.Dictionary.fromLists(labelNameList, labelValList)
        labelvaluesInFilteredRun = tableWithFeaturesToRunWith.aggregate_array(propNameWithNumLabelsToPredictOn).distinct()
        # The regex in replace() is "JS version" - https://bobbyhadz.com/blog/javascript-remove-special-characters-from-string
        # The "python version" regex worked in jupyter lab console but not in code editor "[\W\_]"
        # Using the JS version since it has to run on server.
        labelnamesInFilteredRunWithProbSuffix = labelvaluesInFilteredRun.map(lambda v: ee.String("prob_").cat(labelNameList.getString(ee.Number(v).subtract(1)).replace("[^a-zA-Z0-9_]", "", "gi")))

        labelsSortedByValues = labelnamesInFilteredRunWithProbSuffix.sort(labelvaluesInFilteredRun)
        predictedProbsMultiBand = featureRasterPredicted.arrayFlatten([labelsSortedByValues]).float()

        # Find top-1 label: label with the maximum probability (decision rule)
        # Sample at labeled points to get predicted labels, for train and test set.
        top1Label = featureRasterPredicted.arrayArgmax().arrayFlatten([["top1LabelIndex"]]).uint8()
        top1LabelNum = top1Label.remap(ee.List.sequence(0, labelvaluesInFilteredRun.size().subtract(1)), labelvaluesInFilteredRun.sort()) \
            .rename("top1LabelNum")

        featureRasterPredictions = predictedProbsMultiBand.addBands(top1LabelNum).set(dict( \
            rasterName = "featureRasterPredicted", \
            classifierSchemaFeaturesUsed = trainedModel.schema().join(","), \
            classifierUsed = classifierOptions.get("classifier"), \
            numClassifierTrees = classifierOptions.get("numTrees"), \
            zoneEncoding = featuresOptions.get("zoneEncodingMode"), \
            fractionOfPointsForTrain = classifierOptions.get("trainFraction")))

        # Model prediction accuracy against labeled points table calculations
        # done separately by doing .classify() on the table itself.
        # Sampling predicted raster to pick up predicted label seems prone to
        # breaking the export, it's expensive.
        labelsPredictedForAllPoints = tableWithFeaturesToRunWith.classify(trainedModel)

        # Find top1 label for each point based on classification probabilities
        # Calculate classification performance metrics
        # Save both predicted labels & probs for all points, and performance metrics, separately
        # But first, join the predicted table with master table to pick up string labels column
        joinToMatchPointInMasterTable = ee.Join.saveBest(** {"matchKey": "matchingPoint", "measureKey": "dist"})
        matchPointFilter = ee.Filter.withinDistance(** {"distance": 60, "leftField": ".geo", "rightField": ".geo"})
        labelsPredictedForAllPointsAndLabelPropJoined = joinToMatchPointInMasterTable.apply(
            labelsPredictedForAllPoints,
            pointsWithAllFeatures.select([propNameWithLabels]),
            matchPointFilter)
        allPointsWithTop1Class = labelsPredictedForAllPointsAndLabelPropJoined.map(findtop1LabelNum)

        # Calculate classifier accuracies and performance assessments, for
        # all points ("global"), train and test fractions.
        globalAccSummary = calcClassifierAccuracies(allPointsWithTop1Class, "all")
        trainAccSummary  = calcClassifierAccuracies(allPointsWithTop1Class.filter(trainPtsSplitFilter), "train")
        testAccSummary   = calcClassifierAccuracies(allPointsWithTop1Class.filter(trainPtsSplitFilter.Not()), "test")
        # Repeat these calculations, after collapsing all non-ONE classes into
        # a single class
        labelsList_nonONEsSubset = lulcPalsarHarmnCategoriesInPtsList_nonONEsSubset
        nonONEsNum = 1
        labelRemappedValList = [i for i in range(1, len(labelsList)+1)]
        for l in labelsList_nonONEsSubset:
            labelRemappedValList[labelsList.index(l)] = nonONEsNum
        allPointsWithTop1ClassNonONEsCollapsed = allPointsWithTop1Class \
            .remap(** {
                 "lookupIn": labelValList,
                 "lookupOut": labelRemappedValList,
                 "columnName": propNameWithNumLabelsToPredictOn}) \
            .remap(** {
                 "lookupIn": labelValList,
                 "lookupOut": labelRemappedValList,
                 "columnName": "top1LabelNum"})
        globalNonONEsCollapsedAccSummary = calcClassifierAccuracies(allPointsWithTop1ClassNonONEsCollapsed, "all_nonONEsCollapsed")
        trainNonONEsCollapsedAccSummary  = calcClassifierAccuracies(allPointsWithTop1ClassNonONEsCollapsed.filter(trainPtsSplitFilter), "train_nonONEsCollapsed")
        testNonONEsCollapsedAccSummary   = calcClassifierAccuracies(allPointsWithTop1ClassNonONEsCollapsed.filter(trainPtsSplitFilter.Not()), "test_nonONEsCollapsed")

        accuracyFeatColl = ee.FeatureCollection([ \
            globalAccSummary,                 trainAccSummary,                 testAccSummary, \
            globalNonONEsCollapsedAccSummary, trainNonONEsCollapsedAccSummary, testNonONEsCollapsedAccSummary])

        if resultNewFolderName == None:
            assetFolderName = existingResultFolderName
        else:
            assetFolderName = resultNewFolderName

        if startFreshExport == True:
            # areaOfInterest.geometry()
            probBandsPyramidPol = ee.Dictionary.fromLists(** {
                "keys": predictedProbsMultiBand.bandNames(),
                "values": ee.List.repeat("mean", predictedProbsMultiBand.bandNames().size())})
            top1LabelBandPyramidPol = ee.Dictionary.fromLists(** {
                "keys": top1LabelNum.bandNames(),
                "values": ee.List.repeat("sample", top1LabelNum.bandNames().size())})
            ee.batch.Export.image.toAsset(** {
                'image': featureRasterPredictions,
                'description': "prediction",
                'assetId': assetFolder + assetFolderName + "/" + "prediction",
                'scale': 30,
                'region': reg.geometry(),
                'maxPixels': 1e12,
                "pyramidingPolicy": {".default": "sample"}}).start()
            # ""Dictionary" object does not have evaluate method" error
            # probBandsPyramidPol.combine(top1LabelBandPyramidPol).evaluate(lambda dict: \
            #     ee.batch.Export.image.toAsset(** {
            #         'image': featureRasterPredictions,
            #         'description': "prediction",
            #         'assetId': assetFolder + assetFolderName + "/" + "prediction",
            #         'scale': 30,
            #         'region': reg.geometry(),
            #         'maxPixels': 1e12,
            #         'pyramidingPolicy': dict(dict)}).start()
            # )
            # For long list of features, this getinfo gives user memory exceeded error
            # ee.batch.Export.image.toAsset(** {
            #     'image': featureRasterPredictions,
            #     'description': "prediction",
            #     'assetId': assetFolder + assetFolderName + "/" + "prediction",
            #     'scale': 30,
            #     'region': reg.geometry(),
            #     'maxPixels': 1e12,
            #     'pyramidingPolicy': probBandsPyramidPol.combine(top1LabelBandPyramidPol).getInfo()}).start()
            ee.batch.Export.table.toAsset(** {
                'collection': accuracyFeatColl,
                'description': 'accuracyFeatColl',
                'assetId': assetFolder + assetFolderName + "/" + 'accuracyFeatColl'}).start()
            # .select(labelsSortedByValues.cat(ee.List(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOn, propNameWithLabels, 'top1LabelNum', 'top1Label'])))
            ee.batch.Export.table.toAsset(** {
                'collection': allPointsWithTop1Class,
                'description': 'pointsTablePredicted',
                'assetId': assetFolder + assetFolderName + "/" + 'pointsTablePredicted'}).start()

    return featureRasterPredictions
