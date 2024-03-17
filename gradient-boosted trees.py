from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import GBTClassifier

spark = SparkSession.builder.appName("CancerDiagnosis").getOrCreate()
# Load data
df = spark.read.csv("project3_data.csv", header=True, inferSchema=True)

# Prepare features and labels
feature_columns = ['Radius_mean', 'Texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                   'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
                   'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                   'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                   'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                   'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
label_col = 'diagnosis'
df = df.withColumn(label_col, df[label_col].cast(StringType()))

# Create a feature vector
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
label_indexer = StringIndexer(inputCol=label_col, outputCol="label")

# Create a GBT classifier
gbt_classifier = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)

# Create a pipeline
pipeline = Pipeline(stages=[vector_assembler, label_indexer, gbt_classifier])

# Split the data into training and testing sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Area under ROC curve (GBT): {auc}")

# Additional metrics
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = multi_evaluator.evaluate(predictions)
print(f"Weighted Precision (GBT): {precision}")

multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = multi_evaluator.evaluate(predictions)
print(f"Weighted Recall (GBT): {recall}")

multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = multi_evaluator.evaluate(predictions)
print(f"F1-Score (GBT): {f1_score}")

# Create a parameter grid for hyperparameter tuning
param_grid = ParamGridBuilder().addGrid(gbt_classifier.maxDepth, [3, 5, 7]).build()

# Create a cross-validator
cross_validator = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# Fit the cross-validator to the data
cv_model = cross_validator.fit(train_data)

# Get the best model from cross-validation
best_model = cv_model.bestModel

# Make predictions with the best model
best_predictions = best_model.transform(test_data)

# Evaluate the best model
best_auc = evaluator.evaluate(best_predictions)
print(f"Area under ROC curve (Best GBT Model): {best_auc}")

# Additional metrics for the best model
best_precision = multi_evaluator.evaluate(best_predictions)
print(f"Weighted Precision (Best GBT Model): {best_precision}")

best_recall = multi_evaluator.evaluate(best_predictions)
print(f"Weighted Recall (Best GBT Model): {best_recall}")

best_f1_score = multi_evaluator.evaluate(best_predictions)
print(f"F1-Score (Best GBT Model): {best_f1_score}")

# Feature importances for the best model
best_feature_importances = best_model.stages[-1].featureImportances
print("Feature Importances (Best GBT Model):")
for feature, importance in zip(feature_columns, best_feature_importances):
    print(f"{feature}: {importance}")

# Stop the Spark session
spark.stop()