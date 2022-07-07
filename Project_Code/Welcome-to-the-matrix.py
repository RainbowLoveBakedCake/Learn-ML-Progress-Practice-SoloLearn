#References : https://code.sololearn.com/cA0a10A6A9A1/?ref=app
#default from question
tp, fp, fn, tn = [int(x) for x in input().split()]

#type the formula
total = tp + fp + fn + tn
accuracy = (tp+tn)/total
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score_before_divided = (2 * precision * recall)
real_f1_score = (f1_score_before_divided)/(precision + recall)

#formatting based on test case
accuracy = "{:.4g}".format(accuracy)
precision = "{:.4g}".format(precision)
recall = "{:.4g}".format(recall)
real_f1_score = "{:.4g}".format(real_f1_score)

#print all formula
print(accuracy)
print(precision)
print(recall)
print(real_f1_score)

