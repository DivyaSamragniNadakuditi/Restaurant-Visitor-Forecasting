# Restaurant-Visitor-Forecasting
<h2> Goal</h2>
To predict the number of future visitors a restaurant will receive using Data mining techniques. We further use data visualization tool Tabulae to visualize the data result obtained

Opportunity : Arrangement of staff, To get necessary ingredients

This information will help restaurants be much more efficient and allow them to focus on creating an enjoyable dining experience for their customers.This is a Time-series forecasting problem centered around restaurant visitors

<h2> DATASET DESCRIPTION</h2>
The Data Set chosen is Insightful and Vast USA Statistics.
Link:  https://www.kaggle.com/goldenoakresearch/us-acs-mortgage-equity-loans-rent-statistics/kernels
The data was collected from two sites: Hot Pepper Gourmet, AirREGI/Restaurant Board(air). The first site is used by users to search for the restaurant and make reservations online whereas the second site is used as a cash register system and a reservation control.

The data is extracted in the form of 8 files. The file air_visit_data.csv  has the historic visit data of the air restaurants. The file air_reserve.csv has the data of reservations made through air system and similarly hpg_reserve.csv has the reservation data made through hpg systems.It has around 294K rows and about 80 columns. Using this Data set we can compare, analyze  and draw conclusions about  Population, Age, Income, Debt, Mortgage, Home loan, Gender, Land and Water availability, Marital Status.

Picture1
![Image 1](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture1.png)

<h2> METHODS</h2>
Gradient Boosting Regressor:
Gradient Boosting is basically about "boosting" many weak predictive models into a strong one, in the form of ensemble of weak models. Here, a weak predict model can be any model that works just a little better than random guess.

XGBRegressor:
XGBoost (Extreme Gradient Boosting) belongs to a family of boosting algorithms and uses the gradient boosting (GBM) framework at its core. It is an optimized distributed gradient boosting library
Neural Network:
Neural Networks is another powerful method in which we have multiple layers of network from input to output. It is used to model non-linear relations.

<h2> EVALUATION</H2>

![Image 2](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture2.png)
Gradient Boosting Regressor - Actual    and predicted output. RMSE: 0.3635

![Image 3](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture3.png)
XGBRegressor - Actual and predicted output. RMSE: 0.3502

![Image 4](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture4.png)
Neural Network - Actual and predicted output. RMSE: 0.4861

<h2>Result</h2>
Here is a screenshot of the Results generated
![Image 5](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture5.png)

<h2> Data Visualization</h2>
Here are the data visualization Screenshots generated using Tableau
![Image 6](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture6.png)
Above figure is a Visualization of visitors and Genre in  AirREGI site 

![Image 7](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture7.png)
Above figure is a Visualization of Areas and number of visitors through AirREGI site

![Image 8](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture8.png)
Above figure is a Visualization of Areas and Genre with number of visitors through AirREGI site 

![Image 9](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture9.png)
Above figure is a Visualization of Genre with  number of visitors through Hot Pepper Gourmet site .

![Image 10](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture10.png)
Above figure is a Visualization of Areas and Genre with  number of visitors through Hot Pepper Gourmet site .

![Image 11](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture11.png)
Above figure is a Visualization of Areas and number of visitors through Hot Pepper Gourmet site.

![Image 12](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/images/Picture12.png)
Above figure is a Visualization of Areas where the restaurants are located.

<h2>Poster and Technical Paper</h2>
Here is the Poster created for this project:
![Image 13](https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/Poster.jpg)

The technical paper is present at 
https://github.com/DivyaSamragniNadakuditi/Restaurant-Visitor-Forecasting/blob/master/RESTAURANT%20VISITOR%20FORECASTING%20DATA%20ANALYTICS%20USING%20DATA%20MINING%20TECHNIQUES.pdf
