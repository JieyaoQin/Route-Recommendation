from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
lines = spark.read.text("/home/jieyao/.linuxbrew/Cellar/spark-2.2.0-bin-hadoop2.7/data/mllib/als/rankings.csv").rdd
parts = lines.map(lambda row: row.value.split(","))
routeRankingRDD = parts.map(lambda p: Row(routeID = int(p[0]), userID = int(p[1]), ranking = int(p[2]))
routeRankings = spark.createDataFrame(routeRankingRDD)
#routeRankings.show()
(training, test) = routeRankings.randomSplit([0.8, 0.2])
als = ALS(maxIter = 5, regParam = 0.01, implicitPrefs = True, userCol = "userID", itemCol = "RouteID", ratingCol = "ranking", coldStartStrategy = "drop")
model = als.fit(training)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error =" + str(rmse))
#generate only top 5 choices
userRecs = model.recommendForAllUsers(5)
userRecs.show(5, False)

#for a single user
#query by userID in the userRecs
def give_recs(user_id):
	UserRecs.createOrReplaceTempView("res")
	routes = spark.sql("SELECT recommendations FROM res WHERE userID = " + user_id)
	recs = routes.rdd.map(lambda p: "recommendation: " + p.recommendations).collect()
	for route in recs:
		print("Recommendations for user " + str(user_id) + route)