import graphlab as gl

products=gl.SFrame('amazon_baby.gl/')
#print products
products['word_count']=gl.text_analytics.count_words(products['review'])
#print products['word_count']
#products['name'].show()
girafferiv=products[products['name']=='Vulli Sophie the Giraffe Teether']
#print len(girafferiv)
products=products[products['rating']!=3]
products['sentiment']=products['rating']>=4
#print products.head()

train_data, test_data=products.random_split(0.8, seed=0)
sentiment_model=gl.logistic_classifier.create(train_data, target='sentiment', features=['word_count'], validation_set=test_data)
sentiment_model.evaluate(test_data, metric='roc_curve')

girafferiv['predicted_sentiment']=sentiment_model.predict(girafferiv, output_type='probability')
girafferiv=girafferiv.sort('predicted_sentiment', ascending=False)
print girafferiv.head()
print girafferiv[0]['review'] #most positive review
print girafferiv[-1]['review'] #most negative review