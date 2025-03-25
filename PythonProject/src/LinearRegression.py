import math

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.textpath import text_to_path




class LinearRegression:

    def __init__(self, parameters_dir, train_data, test_data):  # 'self' is the instance
        self.output_sd = None
        self.output_mean = None
        with open(parameters_dir, "r") as file:
            params = yaml.safe_load(file)

        # Assign values to instance attributes using self
        self.batch_size = int(params["model"]["batch_size"])
        self.learning_rate = float(params["model"]["learning_rate"])
        self.param_count = int(params["model"]["param_count"])
        self.epochs = int(params["model"]["epochs"])


        self.train_dataset = train_data
        self.test_dataset = test_data

        # Initialize weights after param_count is set
        self.m_regression = [0.0] * self.param_count
        self.b_regression = 45 #changed



    def model_prediction(self, inputfun):
        prediction = self.b_regression
        for i in range(self.param_count):
            prediction += inputfun.iat[i] * self.m_regression[i]
        return prediction


    def partial_derivative_weights(self, weight_idx, curr_chunk):
        partial_derivative_sum = 0
        if len(curr_chunk) == 0:
            return 0

        for j in range(len(curr_chunk)):
            curr_features = curr_chunk.iloc[j, 0: self.param_count]
            curr_outcome = curr_chunk.iloc[j, self.param_count]
            curr_prediction = self.model_prediction(curr_features)
            curr_x = curr_chunk.iloc[j, weight_idx]
            partial_derivative_sum += 2* (curr_prediction - curr_outcome) * curr_x

        derivative_avg = partial_derivative_sum/len(curr_chunk)
        return derivative_avg

    def partial_derivative_bias(self, curr_chunk):
        if (len(curr_chunk) == 0):
            return 0

        partial_derivative_sum = 0
        for j in range(len(curr_chunk)):

            curr_features = curr_chunk.iloc[j, 0: self.param_count]
            curr_outcome = curr_chunk.iloc[j, self.param_count]
            curr_prediction = self.model_prediction(curr_features)

            partial_derivative_sum += 2* (curr_prediction - curr_outcome)

        derivative_avg = partial_derivative_sum/len(curr_chunk)
        return derivative_avg


    def calculate_loss_squared(self, curr_chunk):
        if (len(curr_chunk) == 0):
            return 0
        total_error_squared = 0
        for j in range(len(curr_chunk)):
            curr_features = curr_chunk.iloc[j, 0:self.param_count]
            curr_outcome = curr_chunk.iloc[j, self.param_count]
            curr_prediction = self.model_prediction(curr_features)
            error = curr_prediction - curr_outcome
            total_error_squared += error * error

        return total_error_squared


    def normalize(self,dataset_norm, column, is_output):

        mean = 0.0
        standard_deviation = 0.0
        entry_count = 0
        for i in range(len(dataset_norm)):
            mean += dataset_norm[column].iloc[i]
            entry_count += 1
        mean /= entry_count
        print("mean:" + str(mean))
        for i in range(0, len(dataset_norm), 1):
            current = dataset_norm[column].iloc[i]
            standard_deviation += (current - mean) * (current - mean) / entry_count

        standard_deviation = math.sqrt(standard_deviation)
        print("sd:" + str(standard_deviation))

        dataset_norm[column] = (dataset_norm[column] - mean) / standard_deviation

        if (is_output):
            self.output_mean = mean
            self.output_sd = standard_deviation


    def invert_normal(self,val):
        return  val * self.output_sd+ self.output_mean


    def train(self):
        epoch_counts = []
        errors = []



        print(self.train_dataset.head())

        for epoch_count in range(self.epochs):
            print("current epoch:"+str(epoch_count))
            epoch_total_loss = 0
            data_entries_cnt = 0

            for start in range(0,len(self.train_dataset),self.batch_size):
                chunk = self.test_dataset.iloc[start: start + self.batch_size]
                if (len(chunk) == 0):
                    break
                chunk_loss = self.calculate_loss_squared(chunk)
                data_entries_cnt += len(chunk)

                for i in range(self.param_count):
                    self.m_regression[i] -= self.partial_derivative_weights(i,chunk) * self.learning_rate

                self.b_regression -= self.partial_derivative_bias(chunk) * self.learning_rate
                epoch_total_loss += chunk_loss

            avg_loss = epoch_total_loss/data_entries_cnt
            epoch_counts.append(epoch_count)
            errors.append(avg_loss)
            print("average_loss:" + str(avg_loss))


        plt.plot(epoch_counts,errors)
        plt.title("Error Plot")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.show()


    def evaluate_model(self):
        total_error = 0
        total_count = 0
        for start in range(0,len(self.test_dataset), self.batch_size):
            chunk = self.test_dataset.iloc[start: start+self.batch_size]
            for j in range(len(chunk)):
                curr_features = chunk.iloc[j, 0:self.param_count]
                curr_outcome_normal = chunk.iloc[j, self.param_count]
                curr_prediction_normal = self.model_prediction(curr_features)

                curr_outcome = curr_outcome_normal
                curr_prediction = curr_prediction_normal

                error = math.fabs(curr_prediction-curr_outcome)
                total_count+=1
                total_error+=error

        return total_error*1.0/total_count

    def getB(self):
        return self.b_regression
    def getM(self):
        return self.m_regression
