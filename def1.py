#!/usr/bin/env python
# coding: utf-8

# In[1]:

def prepare_data(data):            
    if data is None:
        return None
    
   
    

    def only_float(price):
        try:
            x = re.findall(r'[-+]?\d*\.\d+|\d+', price)
            if x:
                return float(x[0])
            else:
                return np.nan
        except:
            return np.nan

    def only_int(price):
        x=''
        try:
            if re.search('[,]',price):
                price_lst=price.split(',')
            else:
                price_lst=price.split()
            for i in price_lst:
                x+=i
            x=re.findall('[0-9]+',x)
            x=float(x[0])
            return x
        except:
            return np.nan

    def no_punc(text):
        # Function to remove punctuation from a string
        import string
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def format_enter(enter):
        if re.search(r'[0-9]+-[0-9]',enter):
            enter=datetime.strptime(enter, '%Y-%m-%d %H:%M:%S').date()
            diff=enter-datetime.now().date()
            if diff.days<183:
                enter='less_then_6_months'
            elif diff.days>183 and diff.days<365:
                enter='months_6_12'
            else:
                enter='above_year'
        elif enter=='גמיש':
            enter='flexible'
        elif enter=='מיידי':
            enter='less_then_6_months'
        else:
            enter='not_defined'
        return enter

    def check_values(x):
        values_to_check = ['True', 'יש מעלית', 'יש', 'yes','כן','yes','יש חנייה','יש מחסן','יש מיזוג אויר','יש מרפסת','יש ממ"ד','נגיש','נגיש לנכים']
        if x in values_to_check:
            return 1
        else:
            return 0 
        
        

    data.rename(columns=lambda x: x.replace(' ', ''), inplace=True) 
    #Shuffle the rows randomly 
    #data = data.sample(frac=1, random_state=42).reset_index(drop=True).copy()

        
        
        
        
        
        

    data.loc[:, 'Area'] = data['Area'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'publishedDays'] = data['publishedDays'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'room_number'] = data['room_number'].astype('str').apply(only_float).astype('float')
    data.loc[:, 'floor'] = data['floor_out_of'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'num_of_images'] = data['num_of_images'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'total_floors'] = data['floor_out_of'].astype('str').apply(lambda x: x.split()[-1] if x.split()[-1].isdigit() else    0)
    
    data.loc[:, 'type'] = data['type'].astype('str').apply(no_punc)
    data.loc[:, 'Street'] = data['Street'].astype('str').apply(no_punc)
    data.loc[:, 'city_area'] = data['city_area'].astype('str').apply(no_punc)
    data.loc[:, 'furniture'] = data['furniture'].astype('str').apply(no_punc)
    data.loc[:, 'description'] = data['description'].astype('str').apply(no_punc)
    
    
    data.loc[:, 'condition'] = data['condition'].astype('str')
    data.loc[:, 'City'] = data['City'].str.replace(r'^\s+|\s+$', '', regex=True).astype('str')
    data.loc[:, 'City'] = data['City'].str.replace('נהרייה', 'נהריה')

   
    data.loc[:, 'hasElevator'] = data['hasElevator'].astype('str').apply(check_values)
    data.loc[:, 'hasParking'] = data['hasParking'].astype('str').apply(check_values)
    data.loc[:, 'hasStorage'] = data['hasStorage'].astype('str').apply(check_values)
    data.loc[:, 'hasAirCondition'] = data['hasAirCondition'].astype('str').apply(check_values)
    data.loc[:, 'hasBalcony'] = data['hasBalcony'].astype('str').apply(check_values)
    data.loc[:, 'hasMamad'] = data['hasMamad'].astype('str').apply(check_values)
    data.loc[:, 'handicapFriendly'] = data['handicapFriendly'].astype('str').apply(check_values)
    data.loc[:, 'hasBars'] = data['hasBars'].astype('str').apply(check_values)
    
    data.loc[:, 'entrance_date'] = data['entranceDate'].astype('str').apply(format_enter)

     # Drop unnecessary columns
    data = data.drop(columns=['floor_out_of', 'number_in_street', 'entranceDate']).copy()
    #data= data[data['price'] > 0]
    data=data.dropna(subset=['price'])
    return data

