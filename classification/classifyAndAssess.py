import ee
import math
import configparser as cp
import io
import ast

from areaOfInterestMask import semiarid as sa
from featuresForModeling import generateFeatures as gf
from utils.trees import TreeParser as tp

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
    lulcPalsarHarmnCategoriesColName = "label_2024"
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
            ee.batch.Export.image.toAsset(** {
                'image': featureRasterPredictions,
                'description': "prediction_l2Flat",
                'assetId': assetFolder + assetFolderName + "/" + "prediction_l2Flat",
                'scale': 30,
                'region': reg.geometry(),
                'maxPixels': 1e12,
                "pyramidingPolicy": {".default": "sample"}}).start()
            ee.batch.Export.table.toAsset(** {
                'collection': accuracyFeatColl,
                'description': 'accuracyFeatColl_l2Flat',
                'assetId': assetFolder + assetFolderName + "/" + 'accuracyFeatColl_l2Flat'}).start()
            ee.batch.Export.table.toAsset(** {
                'collection': allPointsWithTop1Class,
                'description': 'pointsTablePredicted_l2Flat',
                'assetId': assetFolder + assetFolderName + "/" + 'pointsTablePredicted_l2Flat'}).start()

    return featureRasterPredictions

def trainAndPredictNestedLabels(tableWithFeatures, propNameWithNumLabelsToPredictOn, featuresComposite, allLabelnamesAndTheirNumLabels, coarseLevelLabelName, classifierOptions, featuresOptions, resultNewFolderName, returnExisting, startFreshExport):
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

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))
    configCore = config["CORE"]
    assetFolder     = configCore.get('assetFolderWithFeatures')
    lulcPalsarHarmnCategoriesColName = "label_palsarHarmonised"
    configClassify = config["CLASSIFICATION-TRAIN&PREDICT"]
    existingResultFolderName = configClassify.get('existingResultFolderForPredictionsAndAccuracies')
    fullRoiForClassification  = ee.Feature(ee.FeatureCollection(configCore.get('fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

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

    if returnExisting == True:
        featureRasterPredictions = ee.Image(assetFolder + existingResultFolderName + '/prediction')
    else:
        # Split points into train and test sets. Train on train set.
        trainPtsSplitFilter = ee.Filter.lte('random', classifierOptions.get("trainFraction"))
        trainingFraction = tableWithFeatures.filter(trainPtsSplitFilter)
        testingFraction  = tableWithFeatures.filter(trainPtsSplitFilter.Not())
        classifier = selectClassifier(classifierOptions)
        trainedModel = classifier.train(** { \
            'features': trainingFraction, \
            'classProperty': propNameWithNumLabelsToPredictOn, \
            'inputProperties': featuresOptions.get("names"),
        }).setOutputMode('MULTIPROBABILITY')

        # Predict over the full feature raster to produce prediction probabilities.
        featureRasterPredicted = ee.Image(featuresComposite.classify(trainedModel, "classProbabilities"))

        # Convert the array pixel image into multiband image, where band names correspond to the label names
        # First, sort NameList by ValList; NameList and ValList must be arranged such that
        # name-val correspondence is correct
        labelnamesInFilteredRun = allLabelnamesAndTheirNumLabels.keys()
        labelcodesInFilteredRun = allLabelnamesAndTheirNumLabels.values()
        # labelvaluesInFilteredRun = tableWithFeaturesToRunWith.aggregate_array(propNameWithNumLabelsToPredictOn).distinct()
        # The regex in replace() is "JS version" - https://bobbyhadz.com/blog/javascript-remove-special-characters-from-string
        # The "python version" regex worked in jupyter lab console but not in code editor "[\W\_]"
        # Using the JS version since it has to run on server.
        labelnamesInFilteredRunWithProbSuffix = labelnamesInFilteredRun.map(lambda n: ee.String("prob_").cat(ee.String(n).replace("[^a-zA-Z0-9_]", "", "gi")))
        labelsSortedByValues = labelnamesInFilteredRunWithProbSuffix.sort(labelcodesInFilteredRun)

        predictedProbsMultiBand = featureRasterPredicted.arrayFlatten([labelsSortedByValues]).float()

        # Find top-1 label: label with the maximum probability (decision rule)
        # Sample at labeled points to get predicted labels, for train and test set.
        top1Label = featureRasterPredicted.arrayArgmax().arrayFlatten([["top1LabelIndex"]]).uint8()
        top1LabelNum = top1Label.remap(ee.List.sequence(0, labelnamesInFilteredRun.size().subtract(1)), labelcodesInFilteredRun.sort()) \
            .rename("top1LabelNum")

        featureRasterPredictions = predictedProbsMultiBand.addBands(top1LabelNum).set(dict( \
            rasterName = "featureRasterPredicted", \
            classifierSchemaFeaturesUsed = trainedModel.schema().join(","), \
            classifierUsed = classifierOptions.get("classifier"), \
            numClassifierTrees = classifierOptions.get("numTrees"), \
            zoneEncoding = featuresOptions.get("zoneEncodingMode"), \
            fractionOfPointsForTrain = classifierOptions.get("trainFraction")))

        if resultNewFolderName == None:
            assetFolderName = existingResultFolderName
        else:
            assetFolderName = resultNewFolderName

        if startFreshExport == True:
            reg = fullRoiForClassification
            ee.batch.Export.image.toAsset(** {
                'image': featureRasterPredictions,
                'description': "prediction_" + coarseLevelLabelName,
                'assetId': assetFolder + assetFolderName + "/" + "prediction_" + coarseLevelLabelName,
                'scale': 30,
                'region': reg.geometry(),
                'maxPixels': 1e12,
                "pyramidingPolicy": {".default": "sample"}}).start()

    return featureRasterPredictions

def preparePointsAndLabelsForClassification(allPoints, labelTree, coarseLevelNum = 0, coarseLevelLabelName = "Landcover", fineLevelCodeColName = "oneNum", hierarchyMode = "nested", flatHierarchyLevelNum = None):
    fineLevelNum = coarseLevelNum + 1
    labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
    labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
    fineLevelNames = labelNamesL0AndL1AndL2[fineLevelNum]
    fineLevelCodes = labelCodesL0AndL1AndL2[fineLevelNum]

    childNames = labelTree.get_labelInfo_atNode(coarseLevelLabelName, mode = "name")[1]
    childCodes = labelTree.get_labelInfo_atNode(coarseLevelLabelName, mode = "code")[1]

    # In case of flat hierarchy, pass through all points, no ramapping. 
    # Get full list of labels at the req level, ignoring nesting.
    if hierarchyMode == "flat":
        childNamesFlat = labelTree.get_labelInfo_byLevel(mode = "name")[flatHierarchyLevelNum]
        childCodesFlat = labelTree.get_labelInfo_byLevel(mode = "code")[flatHierarchyLevelNum]

        pointsForFineLevelHeirarchClassif = allPoints.randomColumn()
        fineLevelNamesAndTheirCodes = ee.Dictionary.fromLists(childNamesFlat, childCodesFlat)
    # In case of level 0, pass through points and labels. No remapping.
    elif hierarchyMode == "nested" and coarseLevelNum == 0:
        pointsForFineLevelHeirarchClassif = allPoints.randomColumn()
        fineLevelNamesAndTheirCodes = ee.Dictionary.fromLists(childNames, childCodes)
    # Else, for given coarselevel label,
    # remap non-child (finer level) labels to other
    elif hierarchyMode == "nested":
        otherName = "other"
        otherCode = 9999

        fineLevelCodesRemapped = [c if (c in childCodes) else otherCode for c in fineLevelCodes]
        fineLevelNamesRemapped = [c if (c in childNames) else otherName for c in fineLevelNames]
        pointsForFineLevelHeirarchClassif = allPoints \
            .remap(fineLevelCodes, fineLevelCodesRemapped, fineLevelCodeColName) \
            .randomColumn()

        fineLevelNamesExt = childNames + [otherName]
        fineLevelCodesExt = childCodes + [otherCode]
        fineLevelNamesAndTheirCodes = ee.Dictionary.fromLists(fineLevelNamesExt, fineLevelCodesExt)

    return pointsForFineLevelHeirarchClassif, fineLevelNamesAndTheirCodes

def trainAndPredictHierarchical_master(hierarchyProcessingOptions = None, featuresOptions = None, classifierOptions = None, roiForClassification = None, resultNewFolderName = None, returnExisting = False, startFreshExport = False):
    def appendPreviousLevelProbabilities(pt):
        probImage = prevLevelProbs.select(["prob_.*"])
        ptProbSamples = probImage.reduceRegion(** {
            "reducer": ee.Reducer.first(),
            "geometry": pt.geometry(),
            "scale": 30
            # "tileScale": 8
        })
        
        return pt.set(ptProbSamples)


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
    
    lulcLevel1CategoriesColName = hierarchyProcessingOptions.get("coarserLevelLabelColumn")
    lulcPalsarHarmnCategoriesColName = hierarchyProcessingOptions.get("finerLevelLabelColumn")

    lulcLevel1CategoriesInPtsList = ast.literal_eval(configFeaturesAssemble.get("lulcLabelsForONE"))
    existingResultFolderName = config.get('CLASSIFICATION-TRAIN&PREDICT', 'existingResultFolderForPredictionsAndAccuracies')

    # Bring in points with all features table and
    # gather ALL feature rasters, as bands of a single image
    pointsWithAllFeatures = gf.assembleFeatureBandsAndExport(returnExisting = True)
    featuresCompositeToRunWith = gf.assembleAllExistingFeatureRasters() \
        .select(featuresOptions.get("names"))
    print(featuresCompositeToRunWith.bandNames().getInfo())
    
    labelTree = tp().read_from_json("labelHierarchy.json")
    labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
    labelNamesL1 = labelNamesL0AndL1AndL2[1]
    labelNamesL2 = labelNamesL0AndL1AndL2[2]
    labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
    labelCodesL1 = labelCodesL0AndL1AndL2[1]
    labelCodesL2 = labelCodesL0AndL1AndL2[2]

    labelsListCoarse = labelCodesL1
    propNameWithLabelsCoarse = lulcLevel1CategoriesColName
    propNameWithNumLabelsToPredictOnCoarse = propNameWithLabelsCoarse + "Num"

    propNameWithLabelsFine = lulcPalsarHarmnCategoriesColName
    propNameWithNumLabelsToPredictOnFine = propNameWithLabelsFine + "Num"
    
    if hierarchyProcessingOptions.get("hierarchyMode") == "dependent":
        # Level 1, with all labels in level 2 considered together (flat, no hierarchy))
        prevLevelProbs = hierarchyProcessingOptions.get("l1Probabilities") \
            .select(["prob_.*"])
        print(prevLevelProbs.bandNames().getInfo())
        ptsForL2Flat = pointsWithAllFeatures.select(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnFine]) \
            .filter(ee.Filter.notNull(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnFine])) \
            .map(appendPreviousLevelProbabilities)
        featuresCompositeToRunWith = featuresCompositeToRunWith.addBands(prevLevelProbs.select(["prob_.*"]))
        ptsWithFineLevelLabelsPrepped, fineLevelNamesAndCodesPrepped = preparePointsAndLabelsForClassification(ptsForL2Flat, labelTree, 1, labelNamesL1[1], hierarchyMode = "flat", flatHierarchyLevelNum = 2)
        trainAndPredictNestedLabels(ptsWithFineLevelLabelsPrepped, propNameWithNumLabelsToPredictOnFine, featuresCompositeToRunWith, fineLevelNamesAndCodesPrepped, "l2Dep", classifierOptions, featuresOptions, resultNewFolderName, returnExisting, startFreshExport)
        print("hierarchy mode dependent")
    elif hierarchyProcessingOptions.get("hierarchyMode") == "explicit":
        # Run classification, a parent node at a time

        # Level 0
        ptsForL0 = pointsWithAllFeatures.select(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnCoarse]) \
            .filter(ee.Filter.notNull(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnCoarse]))
        ptsWithLevel1LabelsPrepped, level1NamesAndCodes = preparePointsAndLabelsForClassification(ptsForL0, labelTree)
        print("l0 names & codes:", level1NamesAndCodes.getInfo())
        print("l0 hist:", ptsWithLevel1LabelsPrepped.aggregate_histogram(propNameWithNumLabelsToPredictOnCoarse).getInfo())
        trainAndPredictNestedLabels(ptsWithLevel1LabelsPrepped, propNameWithNumLabelsToPredictOnCoarse, featuresCompositeToRunWith, level1NamesAndCodes, "l0", classifierOptions, featuresOptions, resultNewFolderName, returnExisting, startFreshExport)

        # All in level 1, in a loop
        for labL1 in labelNamesL1:
            ptsForL1 = pointsWithAllFeatures.select(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnFine]) \
                .filter(ee.Filter.notNull(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnFine]))
            ptsWithFineLevelLabelsPrepped, fineLevelNamesAndCodesPrepped = preparePointsAndLabelsForClassification(ptsForL1, labelTree, 1, labL1, propNameWithNumLabelsToPredictOnFine)
            print("l1 names & codes:", fineLevelNamesAndCodesPrepped.getInfo())
            print("l1 hist:", ptsWithFineLevelLabelsPrepped.aggregate_histogram(propNameWithNumLabelsToPredictOnFine).getInfo())

            trainAndPredictNestedLabels(ptsWithFineLevelLabelsPrepped, propNameWithNumLabelsToPredictOnFine, featuresCompositeToRunWith, fineLevelNamesAndCodesPrepped, labL1, classifierOptions, featuresOptions, resultNewFolderName, returnExisting, startFreshExport)

        print("hierarchy mode explicit")

    return featuresCompositeToRunWith.bandNames().getInfo()

def calcCoarserLevelLabelsAndAllAccuracies(expHierL1Probs, expHierL2NononeProbs, expHierL2OneProbs, depHierL2Probs, implHierL2Probs, resultNewFolderName):
    def addPredLabels(allgrids, labelImageToSample):
        def sampleLabelsPerGrid(grid):
            pointsInGrid = groundTruthPtsRand.filterBounds(grid.geometry())
            samples = labelImageToSample.reduceRegions(** {
                "reducer": ee.Reducer.first(),
                "collection": pointsInGrid,
                "scale": 30})
            return samples
        return allgrids.map(sampleLabelsPerGrid)
    def calcClassifierAccuracies(labeledTable, gtLabelCol, predLabelCol, labelNumValsOrder, scopeTag):
        predictionAssessment = labeledTable.errorMatrix(** {
            "actual": gtLabelCol,
            "predicted": predLabelCol,
            "order": labelNumValsOrder})
        accuraciesSummary = dict( \
            scope = scopeTag, \
            errMatrixAccuracy = predictionAssessment.accuracy(), \
            errMatrixKappa = predictionAssessment.kappa(), \
            errMatrixConsumersAccuracy = predictionAssessment.consumersAccuracy().project([1]).toList().join(","), \
            errMatrixProducersAccuracy = predictionAssessment.producersAccuracy().project([0]).toList().join(","), \
            errMatrixNameAndOrderOfRowsAndCols = predictionAssessment.order().join(","))
        dummyPoint = ee.Geometry.Point([-65.05351562500002, 37.24376726222654])
        accuraciesFeat = ee.Feature(dummyPoint, accuraciesSummary)
        return accuraciesFeat
    
    def produceAllLabelsAndProbsGivenFullL2ResultRaster(l2ResultRaster):
        l1ProbsOne = l2ResultRaster.select(probBandNamesL2One) \
            .reduce(ee.Reducer.sum()) \
            .rename("prob_one")
        l1ProbsNonone = l2ResultRaster.select(probBandNamesL2Nonone) \
            .reduce(ee.Reducer.sum()) \
            .rename("prob_nonone")
        l1Probs = l1ProbsOne.addBands(l1ProbsNonone)

        l1Top1 = l2ResultRaster.select('top1LabelNum').remap(** {
            "from": oneGroupNums + nononeGroupNums,
            "to": [oneNum]*len(oneGroupNums) + [nononeNum]*len(nononeGroupNums), 
            "bandName": 'top1LabelNum'}) \
            .rename("l1LabelNum")
        l2Top1 = l2ResultRaster.select(['top1LabelNum'], ['l2LabelNum'])
        allLevelLabels = l1Top1.addBands(l2Top1)

        l1Confidence = l1Probs.toArray().arrayReduce(ee.Reducer.max(), [0]) \
            .arrayFlatten([["probL1Label"]])
        l2Confidence = l2ResultRaster.select(probBandNamesL2Nonone.cat(probBandNamesL2One).sort()) \
            .toArray().arrayReduce(ee.Reducer.max(), [0]) \
            .arrayFlatten([["probL2Label"]])
        allLevelTop1Conf = l1Confidence.addBands(l2Confidence)
        
        depHierStepAllLevelLabelsAllConfsProbs = ee.Image.cat([ \
            allLevelLabels.uint8(), \
            allLevelTop1Conf.multiply(1e4).round().uint16(), \
            l1Probs.multiply(1e4).round().uint16(), \
            l2ResultRaster.select(probBandNamesL2One, probBandNamesL2OneWithL1Prefix).multiply(1e4).round().uint16(), \
            l2ResultRaster.select(probBandNamesL2Nonone, probBandNamesL2NononeWithL1Prefix).multiply(1e4).round().uint16()])
        
        return depHierStepAllLevelLabelsAllConfsProbs

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))
    configCore = config["CORE"]
    assetFolder     = configCore.get('assetFolderWithFeatures')
    origTrainingPoints = gf.assembleFeatureBandsAndExport(returnExisting = True)
    l1LabelNumColName = "labelL1Num"
    l2LabelNumColName = "label_2024Num"
    configClassify = config["CLASSIFICATION-TRAIN&PREDICT"]
    existingResultFolderName = configClassify.get('existingResultFolderForPredictionsAndAccuracies')
    fullRoiForClassification  = ee.Feature(ee.FeatureCollection(configCore.get('fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    labelTree = tp().read_from_json("labelHierarchy.json")
    coarseLevelNum = 1
    fineLevelNum = coarseLevelNum + 1
    labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
    labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
    coarseLevelNames = labelNamesL0AndL1AndL2[coarseLevelNum]
    coarseLevelCodes = labelCodesL0AndL1AndL2[coarseLevelNum]
    fineLevelNames = labelNamesL0AndL1AndL2[fineLevelNum]
    fineLevelCodes = labelCodesL0AndL1AndL2[fineLevelNum]

    oneGroupNums = labelTree.get_labelInfo_atNode("one", mode = "code")[1]
    nononeGroupNums = labelTree.get_labelInfo_atNode("nonone", mode = "code")[1]
    oneNum = labelTree.find_by_name("one").code
    nononeNum = labelTree.find_by_name("nonone").code
    
    # List of class probabilitiy band names, in prep for combining probabilities
    probBandNamesL1 = expHierL1Probs.bandNames().removeAll(['top1LabelNum'])
    probBandNamesL2One    = expHierL2OneProbs.bandNames().removeAll(['top1LabelNum', 'prob_other'])
    probBandNamesL2Nonone = expHierL2NononeProbs.bandNames().removeAll(['top1LabelNum', 'prob_other'])

    # Remove "prob_" prefix
    l1ClassNames = probBandNamesL1.map(lambda n: ee.String(n).slice(5))
    l2OneClassNames = probBandNamesL2One.map(lambda n: ee.String(n).slice(5))
    l2NononeClassNames = probBandNamesL2Nonone.map(lambda n: ee.String(n).slice(5))

    # Class labels and their codes, according to original convention, for each level
    allClassNamesSorted = l2OneClassNames.cat(l2NononeClassNames).sort()
    l1ClassNamesSorted = l1ClassNames.sort()
    l2OneClassNamesSorted = l2OneClassNames.sort()
    l2NononeClassNamesSorted = l2NononeClassNames.sort()
    l2OneClassCodesForSortedLabels = l2OneClassNamesSorted.map(lambda n: allClassNamesSorted.indexOf(ee.String(n)).add(1))
    l2NononeClassCodesForSortedLabels = l2NononeClassNamesSorted.map(lambda n: allClassNamesSorted.indexOf(ee.String(n)).add(1))

    ####### Readjust probabilities ######
    l1OneReadjust = expHierL1Probs.select("prob_one") \
        .multiply(ee.Image(1).subtract(expHierL2OneProbs.select("prob_other")))
    l1NononeReadjust = expHierL1Probs.select("prob_nonone") \
        .multiply(ee.Image(1).subtract(expHierL2NononeProbs.select("prob_other")))
    # standardise (sum to 1) readjusted l1 probs 
    l1Readjust = ee.Image.cat([l1NononeReadjust, l1OneReadjust]) \
        .divide(l1NononeReadjust.add(l1OneReadjust))

    # standardise (sum to 1) readjusted l2 probs
    l2OneReadjust = expHierL2OneProbs.select(probBandNamesL2One) \
        .divide(expHierL2OneProbs.select(probBandNamesL2One).reduce(ee.Reducer.sum()))
    l2NononeReadjust = expHierL2NononeProbs.select(probBandNamesL2Nonone) \
        .divide(expHierL2NononeProbs.select(probBandNamesL2Nonone).reduce(ee.Reducer.sum()))

    ###### Hierarchical multiplicative rule
    # Multiply L1 probabilities with their corresponding set of probabilities from L2
    l1MultL2One = l1Readjust.select('prob_one') \
        .multiply(l2OneReadjust.select(probBandNamesL2One))\
        .rename(l2OneClassNames)
    l1MultL2Nonone = l1Readjust.select('prob_nonone') \
        .multiply(l2NononeReadjust.select(probBandNamesL2Nonone)) \
        .rename(l2NononeClassNames)

    # Merge all these probabilities and calculate their top 1 label
    allProbsInArray = l1MultL2One.addBands(l1MultL2Nonone) \
        .select(allClassNamesSorted) \
        .toArray()
    expHierMultL2 = allProbsInArray.arrayArgmax().arrayFlatten([["top1LabelIndexHei"]]).uint8() \
        .remap(ee.List.sequence(0, len(fineLevelNames)-1), ee.List.sequence(1, len(fineLevelNames)))
    expHierMultL1 = expHierMultL2.remap(** {
        "from": oneGroupNums + nononeGroupNums,
        "to": [oneNum]*len(oneGroupNums) + [nononeNum]*len(nononeGroupNums), 
        "bandName": 'remapped'})
        
    expHierMultAllLevelLabels = ee.Image.cat([expHierMultL1.rename("l1LabelNum"), expHierMultL2.rename("l2LabelNum")])

    # Find prob of top1 label
    expHierMultL2Confidence = allProbsInArray.arrayReduce(ee.Reducer.max(), [0]) \
        .arrayFlatten([["probL2Label"]])
    expHierMultL1NononeProb = ee.Image(0).where(expHierMultL1.eq(nononeNum), l1MultL2Nonone.reduce(ee.Reducer.sum()))
    expHierMultL1OneProb    = ee.Image(0).where(expHierMultL1.eq(oneNum), l1MultL2One.reduce(ee.Reducer.sum()))
    expHierMultL1Confidence = expHierMultL1NononeProb.add(expHierMultL1OneProb).rename("probL1Label")
    
    expHierMultAllLevelTop1Conf = ee.Image.cat([expHierMultL1Confidence, expHierMultL2Confidence])

    # For all levels, combine top 1 labels with their probs, and probs of all labels
    probBandNamesL2OneWithL1Prefix    =    probBandNamesL2One.map(lambda n: ee.String('prob_one_').cat(ee.String(n).slice(5)))
    probBandNamesL2NononeWithL1Prefix = probBandNamesL2Nonone.map(lambda n: ee.String('prob_nonone_').cat(ee.String(n).slice(5)))
    expHierMultAllLevelLabelsAllConfsProbs = ee.Image.cat([ \
        expHierMultAllLevelLabels.uint8(), \
        expHierMultAllLevelTop1Conf.multiply(1e4).round().uint16(), \
        l1Readjust.multiply(1e4).round().uint16(), \
        l1MultL2One.select(l2OneClassNames, probBandNamesL2OneWithL1Prefix).multiply(1e4).round().uint16(), \
        l1MultL2Nonone.select(l2NononeClassNames, probBandNamesL2NononeWithL1Prefix).multiply(1e4).round().uint16()])

    trainFrac = 0.7
    groundTruthPtsRand = origTrainingPoints.select([l1LabelNumColName, l2LabelNumColName]).randomColumn()
    # Create a covering grid over the region covered by points. Sampling by mapping over them goes much faster
    proj = ee.Projection("EPSG:4326").scale(5, 5)
    grids = origTrainingPoints.geometry().bounds().coveringGrid(proj)
    gtPtsExpHierMultPredLabelAdded = addPredLabels(grids, expHierMultAllLevelLabels).flatten()
    
    expHierMultTrainPts = gtPtsExpHierMultPredLabelAdded.filter(ee.Filter.lte('random', trainFrac))
    expHierMultTrainAccL2 = calcClassifierAccuracies(expHierMultTrainPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "expHierMult_l2Train")
    expHierMultTrainAccL1 = calcClassifierAccuracies(expHierMultTrainPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "expHierMult_l1Train")
    expHierMultTestPts = gtPtsExpHierMultPredLabelAdded.filter(ee.Filter.gt('random', trainFrac))
    expHierMultTestAccL2 = calcClassifierAccuracies(expHierMultTestPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "expHierMult_l2Test")
    expHierMultTestAccL1 = calcClassifierAccuracies(expHierMultTestPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "expHierMult_l1Test")
    
    ###### Hierarchical step-wise rule
    # Top 1 label at each level seperately
    l1ProbsInArray = l1Readjust.select(probBandNamesL1, l1ClassNames) \
        .select(l1ClassNamesSorted) \
        .toArray()
    l2OneProbsInArray = l2OneReadjust.select(probBandNamesL2One, l2OneClassNames) \
        .select(l2OneClassNamesSorted) \
        .toArray()
    l2NononeProbsInArray = l2NononeReadjust.select(probBandNamesL2Nonone, l2NononeClassNames) \
        .select(l2NononeClassNamesSorted) \
        .toArray()

    expHierStepL1 = l1ProbsInArray.arrayArgmax().arrayFlatten([["top1LabelIndexL1"]]).uint8() \
        .remap([0, 1], [nononeNum, oneNum], bandName = 'top1LabelIndexL1')
    # remap top 1 index to the code of the corresponding label
    expHierStepL2One = l2OneProbsInArray.arrayArgmax().arrayFlatten([["top1LabelIndexL2One"]]).uint8() \
        .remap(ee.List.sequence(0, l2OneClassNames.length().subtract(1)), l2OneClassCodesForSortedLabels)
    # remap top 1 index to the code of the corresponding label
    expHierStepL2Nonone = l2NononeProbsInArray.arrayArgmax().arrayFlatten([["top1LabelIndexL2Nonone"]]).uint8() \
        .remap(ee.List.sequence(0, l2NononeClassNames.length().subtract(1)), l2NononeClassCodesForSortedLabels)

    # Apply stepwise majority rule to get final class labels
    nononeStepwise = expHierStepL1.eq(nononeNum).selfMask() \
        .multiply(expHierStepL2Nonone)
    oneStepwise = expHierStepL1.eq(oneNum).selfMask() \
        .multiply(expHierStepL2One)
    expHierStepL2 = nononeStepwise.unmask(0).add(oneStepwise.unmask(0))

    expHierStepAllLevelLabels = ee.Image.cat([expHierStepL1.rename("l1LabelNum"), expHierStepL2.rename("l2LabelNum")])

    # Find prob of top1 label
    expHierStepL1Confidence = l1ProbsInArray.arrayReduce(ee.Reducer.max(), [0]) \
        .arrayFlatten([["probL1Label"]])
    l2NononeTop1Prob = l2NononeProbsInArray.arrayReduce(ee.Reducer.max(), [0]) \
        .arrayFlatten([["probL2Label_nonone"]])
    l2OneTop1Prob = l2OneProbsInArray.arrayReduce(ee.Reducer.max(), [0]) \
        .arrayFlatten([["probL2Label_one"]])
    nononeStepwiseProb = expHierStepL1.eq(nononeNum).selfMask() \
        .multiply(l2NononeTop1Prob)
    oneStepwiseProb = expHierStepL1.eq(oneNum).selfMask() \
        .multiply(l2OneTop1Prob)
    expHierStepL2Confidence = nononeStepwiseProb.unmask(0).add(oneStepwiseProb.unmask(0)).rename("probL2Label")

    expHierStepAllLevelTop1Conf = ee.Image.cat([expHierStepL1Confidence, expHierStepL2Confidence])

    # For all levels, combine top 1 labels with their probs, and probs of all labels
    expHierStepAllLevelLabelsAllConfsProbs = ee.Image.cat([ \
        expHierStepAllLevelLabels.uint8(), \
        expHierStepAllLevelTop1Conf.multiply(1e4).round().uint16(), \
        l1Readjust.multiply(1e4).round().uint16(), \
        l2OneReadjust.select(probBandNamesL2One, probBandNamesL2OneWithL1Prefix).multiply(1e4).round().uint16(), \
        l2NononeReadjust.select(probBandNamesL2Nonone, probBandNamesL2NononeWithL1Prefix).multiply(1e4).round().uint16()])

    gtPtsExpHierStepPredLabelAdded = addPredLabels(grids, expHierStepAllLevelLabels).flatten() 
    
    expHierStepTrainPts = gtPtsExpHierStepPredLabelAdded.filter(ee.Filter.lte('random', trainFrac))
    expHierStepTrainAccL2 = calcClassifierAccuracies(expHierStepTrainPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "expHierStep_l2Train")
    expHierStepTrainAccL1 = calcClassifierAccuracies(expHierStepTrainPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "expHierStep_l1Train")
    expHierStepTestPts = gtPtsExpHierStepPredLabelAdded.filter(ee.Filter.gt('random', trainFrac))
    expHierStepTestAccL2 = calcClassifierAccuracies(expHierStepTestPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "expHierStep_l2Test")
    expHierStepTestAccL1 = calcClassifierAccuracies(expHierStepTestPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "expHierStep_l1Test")
    
    ###### Dependent hierarchy
    depHierStepAllLevelLabelsAllConfsProbs = produceAllLabelsAndProbsGivenFullL2ResultRaster(depHierL2Probs)
    gtPtsDepHierPredLabelAdded = addPredLabels(grids, depHierStepAllLevelLabelsAllConfsProbs.select('l1LabelNum', 'l2LabelNum')).flatten() 
    
    depHierTrainPts = gtPtsDepHierPredLabelAdded.filter(ee.Filter.lte('random', trainFrac))
    depHierTrainAccL2 = calcClassifierAccuracies(depHierTrainPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "depHier_l2Train")
    depHierTrainAccL1 = calcClassifierAccuracies(depHierTrainPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "depHier_l1Train")
    depHierTestPts = gtPtsDepHierPredLabelAdded.filter(ee.Filter.gt('random', trainFrac))
    depHierTestAccL2 = calcClassifierAccuracies(depHierTestPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "depHier_l2Test")
    depHierTestAccL1 = calcClassifierAccuracies(depHierTestPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "depHier_l1Test")
    
    ###### Implicit hierarchy
    implHierStepAllLevelLabelsAllConfsProbs = produceAllLabelsAndProbsGivenFullL2ResultRaster(implHierL2Probs)
    gtPtsDepHierPredLabelAdded = addPredLabels(grids, implHierStepAllLevelLabelsAllConfsProbs.select('l1LabelNum', 'l2LabelNum')).flatten() 
    
    implHierTrainPts = gtPtsDepHierPredLabelAdded.filter(ee.Filter.lte('random', trainFrac))
    implHierTrainAccL2 = calcClassifierAccuracies(implHierTrainPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "implHier_l2Train")
    implHierTrainAccL1 = calcClassifierAccuracies(implHierTrainPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "implHier_l1Train")
    implHierTestPts = gtPtsDepHierPredLabelAdded.filter(ee.Filter.gt('random', trainFrac))
    implHierTestAccL2 = calcClassifierAccuracies(implHierTestPts, l2LabelNumColName, 'l2LabelNum', sorted(fineLevelCodes), "implHier_l2Test")
    implHierTestAccL1 = calcClassifierAccuracies(implHierTestPts, l1LabelNumColName, 'l1LabelNum', sorted(coarseLevelCodes), "implHier_l1Test")

    accuracyFeatColl = ee.FeatureCollection( \
        [expHierMultTrainAccL1, expHierMultTestAccL1, expHierMultTrainAccL2, expHierMultTestAccL2, \
         expHierStepTrainAccL1, expHierStepTestAccL1, expHierStepTrainAccL2, expHierStepTestAccL2, \
         depHierTrainAccL1, depHierTestAccL1, depHierTrainAccL2, depHierTestAccL2, \
         implHierTrainAccL1, implHierTestAccL1, implHierTrainAccL2, implHierTestAccL2])
    
    reg = fullRoiForClassification
    outputSuffix = "expHierMult"
    ee.batch.Export.image.toAsset(** {
        'image': expHierMultAllLevelLabelsAllConfsProbs,
        'description': "prediction_" + outputSuffix,
        'assetId': assetFolder + resultNewFolderName + "/" + "prediction_" + outputSuffix,
        'scale': 30,
        'region': reg.geometry(),
        'maxPixels': 1e12,
        "pyramidingPolicy": {".default": "sample"}}).start()
    outputSuffix = "expHierStep"
    ee.batch.Export.image.toAsset(** {
        'image': expHierStepAllLevelLabelsAllConfsProbs,
        'description': "prediction_" + outputSuffix,
        'assetId': assetFolder + resultNewFolderName + "/" + "prediction_" + outputSuffix,
        'scale': 30,
        'region': reg.geometry(),
        'maxPixels': 1e12,
        "pyramidingPolicy": {".default": "sample"}}).start()
    outputSuffix = "depHeir"
    ee.batch.Export.image.toAsset(** {
        'image': depHierStepAllLevelLabelsAllConfsProbs,
        'description': "prediction_" + outputSuffix,
        'assetId': assetFolder + resultNewFolderName + "/" + "prediction_" + outputSuffix,
        'scale': 30,
        'region': reg.geometry(),
        'maxPixels': 1e12,
        "pyramidingPolicy": {".default": "sample"}}).start()
    outputSuffix = "implHeir"
    ee.batch.Export.image.toAsset(** {
        'image': depHierStepAllLevelLabelsAllConfsProbs,
        'description': "prediction_" + outputSuffix,
        'assetId': assetFolder + resultNewFolderName + "/" + "prediction_" + outputSuffix,
        'scale': 30,
        'region': reg.geometry(),
        'maxPixels': 1e12,
        "pyramidingPolicy": {".default": "sample"}}).start()
    ee.batch.Export.table.toAsset(** {
        'collection': accuracyFeatColl,
        'description': 'accuracyFeatColl_' + "expDepImplHier",
        'assetId': assetFolder + resultNewFolderName + "/" + 'accuracyFeatColl_' + "expDepImplHier"}).start()

    return accuracyFeatColl
