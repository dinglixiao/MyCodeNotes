package com.bitscott

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.SparkSession

/**
  * Hello world!
  *
  */
object Application extends App with Logging {

  val conf = new SparkConf().setAppName("app").setMaster("local[2]")
  val spark = SparkSession.builder().config(conf).getOrCreate()

  import org.apache.spark.ml.feature.Tokenizer
  import org.apache.spark.sql.functions._

  val sentenceDataFrame = spark.createDataFrame(Seq(
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
  )).toDF("id", "sentence")

  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  //  val regexTokenizer = new RegexTokenizer()
  //    .setInputCol("sentence")
  //    .setOutputCol("words")
  //    .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

  val countTokens = udf { (words: Seq[String]) => words.length }


  val tokenized = tokenizer.transform(sentenceDataFrame)
  tokenized.select("sentence", "words")
    .withColumn("tokens", countTokens(col("words"))).show(false)

  //
  //  val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
  //  regexTokenized.show(false)
  //
  //  regexTokenized.select("sentence", "words")
  //    .withColumn("tokens", countTokens(col("words"))).show(false)

  val hashingTF = new HashingTF()
    .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(200)

  val featurizedData = hashingTF.transform(tokenized)

  featurizedData.show(false)
}
