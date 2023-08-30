import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pprint
from sklearn import metrics
import random

df = pd.read_csv('milk.csv')
dt_Train, dt_Test = train_test_split(df, test_size=0.3 , shuffle = False)

#tính entropy: I
def find_entropy(df):
  Class = df.keys()[-1]
  values = df[Class].unique()
  entropy = 0
  for value in values:
    prob = df[Class].value_counts()[value]/len(df[Class])
    entropy += -prob * np.log2(prob)
  return float(entropy)


# tính entropy attribute: E
def find_entropy_attribute(df, attribute):
  Class = df.keys()[-1]
  target_values = df[Class].unique()
  attribute_values = df[attribute].unique()
  avg_entropy = 0
  for value in attribute_values:
    entropy = 0
    for value1 in target_values:
      num = len(df[attribute][df[attribute] == value][df[Class] == value1]) #len(df[(df[attribute] == value) & (df[Class] == value1)])
      den = len(df[attribute][df[attribute] == value])
      prob = num/den
      entropy += -prob * np.log2(prob + 0.000001)
    avg_entropy += (den/len(df))*entropy
  return float(avg_entropy)


#Find Winner: Gain
def find_winner(df):
  IG = []
  for key in df.keys()[:-1]:
    IG.append(find_entropy(df) - find_entropy_attribute(df, key))
  return df.keys()[:-1][np.argmax(IG)]
  #np.argmax(IG): trả về chỉ mục có Gain lớn nhất
  #df.keys()[:-1][np.argmax(IG)] truy cập vào danh sách tên cột, loại bỏ cột cuối cùng và trả về tên cột có chỉ mục tương ứng với Information Gain (IG) lớn nhất.


#lấy bảng phụ
def get_subtable(df, attribute, value):
  return df[df[attribute] == value].reset_index(drop = True) #reset_index(drop=True) được sử dụng để thiết lập lại chỉ số của các dòng trong phụ-bảng, bỏ qua chỉ số ban đầu và tạo một chỉ số mới bắt đầu từ 0.


#xây dựng cây
def buildtree(df, tree = None):
  node = find_winner(df)
  attvalue = np.unique(df[node])
  Class = df.keys()[-1]
  if tree is None:
    tree = {} #tạo từ điển rỗng
    tree[node] = {}
  for value in attvalue:
    subtable = get_subtable(df,node,value)
    Clvalue, counts = np.unique(subtable[Class], return_counts = True)  #Clvalue: mảng chứa giá trị duy nhất của cột nhãn, counts: mảng chứa số lần xuất hiện của từng nhãn trong subtable
    if len(counts) == 1:
      tree[node][value] = Clvalue[0]
    else:
      tree[node][value] = buildtree(subtable)
  return tree

tree = buildtree(dt_Train)
print("- CÂY QUYẾT ĐỊNH: ")
pprint.pprint(tree)


#dự đoán nhãn
def predict(inst, tree):
  for node in tree.keys():  #tên node root: ph
    value = inst[node]

    if node in tree and value in tree[node]:  tree = tree[node][value]
    else: tree = random.choice(dt_Test["chat_luong"].unique())
    
    prediction = 0
    if type(tree) is dict:
      prediction = predict(inst, tree)
    else:
      prediction = tree
  return prediction


Y_label = []
for i in range(len(dt_Test)):
  inst = dt_Test.iloc[i,:]
  prediction = predict(inst, tree)
  Y_label.append(prediction)
  
print("\n- NHÃN DỰ ĐOÁN TRÊN TẬP TEST:\n",Y_label)

print("\n- ĐỘ ĐO:\n",metrics.classification_report(dt_Test.iloc[:,-1], Y_label))


#dự đoán nhãn trên 1 mẫu dữ liệu:
mau1 = {"ph": 9, "nhiet_do": 1, "huong_vi": 23, "mui": 7.54, "chat_beo": 0, "mau": 220, "do_trong": 11.3}
print("- Nhãn dự đoán của mau1 : ",predict(mau1, tree))



