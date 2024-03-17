from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder.appName("CancerDiagnosis").getOrCreate()
df = spark.read.csv("project3_data.csv", header=True, inferSchema=True)
feature_columns = ['Radius_mean', 'Texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                   'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
                   'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                   'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                   'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                   'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

label_col = 'diagnosis'
df = df.withColumn(label_col, df[label_col].cast(StringType()))
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
label_indexer = StringIndexer(inputCol=label_col, outputCol="label")
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[vector_assembler, label_indexer, rf_classifier])
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Area under ROC curve: {auc}")
# feature_importances = model.stages[-1].featureImportances
# print("Feature Importances:")
# for i, importance in enumerate(feature_importances):
#     print(f"Feature {feature_columns[i]}: {importance}")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = multi_evaluator.evaluate(predictions)
print(f"Weighted Precision: {precision}")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = multi_evaluator.evaluate(predictions)
print(f"Weighted Recall: {recall}")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = multi_evaluator.evaluate(predictions)
print(f"F1-Score: {f1_score}")
param_grid = ParamGridBuilder().addGrid(rf_classifier.numTrees, [10, 20, 30]).build()
cross_validator = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
cv_model = cross_validator.fit(train_data)

# Extracting probability and label columns from predictions
results = predictions.select(['probability', 'label']).rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))

# Instantiate BinaryClassificationMetrics
metrics = BinaryClassificationMetrics(results)

# Get the ROC curve
roc_curve = metrics.roc()

# Plotting ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(roc_curve[0])):
    fpr[i] = roc_curve[0][i]
    tpr[i] = roc_curve[1][i]
    roc_auc[i] = roc_curve[2][i]

plt.figure(figsize=(8, 8))
plt.plot(fpr[1], tpr[1], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

best_model = cv_model.bestModel
predictions = best_model.transform(df)
print(" ")
precision = multi_evaluator.evaluate(predictions)
print(f"Weighted Precision: {precision}")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = multi_evaluator.evaluate(predictions)
print(f"Weighted Recall: {recall}")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = multi_evaluator.evaluate(predictions)
print(f"F1-Score: {f1_score}")
spark.stop()