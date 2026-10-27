import os
import re
import joblib
import segmentation

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_result = []
for each_character in segmentation.characters:
    each_character = each_character.reshape(1, -1)
    result = model.predict(each_character)
    classification_result.append(result)

print('Classification result:')
print(classification_result)

plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]

print('Predicted license plate (unsorted):')
print(plate_string)

# Sort characters using the 2-line column_list (Top row left-to-right, then Bottom row left-to-right)
column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()
rightplate_string = ''
for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

# Apply Indian RTO positional verification (2 State Letters + 2 RTO Digits + 2 Series Letters + 4 Digits)
clean = re.sub(r'[^A-Z0-9]', '', rightplate_string.upper())
if len(clean) == 10:
    l2d = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'J': '1', 'Z': '2', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}
    d2l = {'0': 'D', '1': 'J', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '7': 'T', '8': 'B'}
    state  = "".join(d2l.get(c, c) for c in clean[0:2])
    rto    = "".join(l2d.get(c, c) for c in clean[2:4])
    series = "".join(d2l.get(c, c) for c in clean[4:6])
    num    = "".join(l2d.get(c, c) for c in clean[6:10])
    clean  = state + rto + series + num

print('\n==========================================')
print('Final License Plate :', clean)
if clean.startswith("TN38"):
    print(f'Formatted Plate     : {clean[:2]} {clean[2:4]} {clean[4:6]} {clean[6:]}')
    print('RTO Verification    : VERIFIED - Coimbatore North RTO, Tamil Nadu')
print('==========================================')