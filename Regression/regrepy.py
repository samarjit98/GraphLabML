import graphlab as gl 
import matplotlib.pyplot as plt 

sales=gl.SFrame('home_data.gl/')
#sales.show(view="Scatter Plot", x="sqft_living", y="price")
train_data, test_data=sales.random_split(0.8, seed=0)
'''
print train_data
print test_data
'''
sqft_model=gl.linear_regression.create(train_data, target='price', features=['sqft_living'])
#print test_data['price'].mean()
print sqft_model.evaluate(test_data)
print sqft_model.get('coefficients')

plt.plot(test_data['sqft_living'], test_data['price'], '.', test_data['sqft_living'], sqft_model.predict(test_data), '-')
plt.show()
print sqft_model.get('coefficients')

my_features=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
my_features_model=gl.linear_regression.create(train_data, target='price', features=my_features)
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

house1=sales[sales['id']=='5309101200']
print house1
print house1['price']
print sqft_model.predict(house1)
print my_features_model.predict(house1)

house2=sales[sales['id']=='1925069082']
print house2
print house2['price']
print sqft_model.predict(house2)
print my_features_model.predict(house2)