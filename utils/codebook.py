categories = {'angle':[str(i) for i in range(18)], 'distance':[str(i) for i in range(10)], 'relX':['at the right of','x-ignored', 'at the left of'], 'relY':['below', 'y-ignored','above'], 'relZ':['behind', 'z-ignored','in front of'], 'relV': ['vertical', 'ignored', 'horizontal'], 'ground':['on the ground', 'ground-ignored']}
codes = ['L-knee', 'R-knee', 'L-elbow', 'R-elbow', # angle 
         'L-elbow vs R-elbow', 'L-hand vs R-hand', 'L-knee vs R-knee', 'L-foot vs R-foot',  # dist
         'L-hand vs L-shoulder', 'L-hand vs R-shoulder', 'R-hand vs R-shoulder', 'R-hand vs L-shoulder', 'L-hand vs R-elbow', 'R-hand vs L-elbow',  # dist
         'L-hand vs L-knee', 'L-hand vs R-knee', 'R-hand vs R-knee', 'R-hand vs L-knee', 'L-hand vs L-foot', 'L-hand vs R-foot', 'R-hand vs R-foot', 'R-hand vs L-foot', # dist
         'L-hand vs R-hand', 'neck vs pelvis', 'L-hand vs L-shoulder', 'R-hand vs R-shoulder', 'L-foot vs L-hip', 'R-foot vs R-hip', # rel-x
         'L-shoulder vs R-shoulder', 'L-elbow vs R-elbow', 'L-hand vs R-hand', 'L-ankle vs neck', 'R-ankle vs neck', 'L-knee vs R-knee', 'L-hip vs L-knee', 'R-hip vs R-knee', 'L-hand vs L-shoulder', 'R-hand vs R-shoulder', 'L-foot vs L-hip', 'R-foot vs R-hip', 'L-wrist vs neck', 'R-wrist vs neck', 'L-hand vs L-hip', 'R-hand vs R-hip',  # rel-X
         'L-shoulder vs R-shoulder', 'L-elbow vs R-elbow', 'L-hand vs R-hand', 'L-knee vs R-knee', 'neck vs pelvis', 'L-hand vs torso', 'R-hand vs torso', 'L-foot vs torso', 'R-foot vs torso',  # rel-Y
         'L-hip vs L-knee', 'R-hip vs R-knee', 'L-knee vs L-ankle', 'R-knee vs R-ankle', 'L-shoulder vs L-elbow', 'R-shoulder vs R-elbow', 'L-elbow vs L-wrist', 'R-elbow vs R-wrist', 'pelvis vs L-shoulder', 'pelvis vs R-shoulder', 'pelvis vs neck', 'L-hand vs R-hand', 'L-foot vs R-foot', # rel-Z
         'L-knee', 'R-knee', 'L-foot', 'R-foot'] # ground

offsets = {'angle': 0, 'distance':18, 'relX':28, 'relY':31, 'relZ':34, 'relV': 37, 'ground':40}
total = 0
curr_id = 0
name_to_id = {}
id_to_name = {}
vq_to_trans = {}
vq_to_cat = {}
vq_to_range = {}

for i in range(70):
    name = codes[i]

    if i < 4:
        key = 'angle'
    elif i < 22:
        key = 'distance'
    elif i < 28:
        key = 'relX'
    elif i < 44:
        key = 'relY'
    elif i < 53:
        key = 'relZ'
    elif i < 66:
        key = 'relV'
    else:
        key = 'ground'
    
    total += len(categories[key])
    vq_to_cat[i] = len(categories[key])
    offset = offsets[key]

    for j in range(len(categories[key])): 
        full = name + " " + categories[key][j] 
        name_to_id[full] = curr_id 
        id_to_name[curr_id] = full 
        temp = vq_to_trans.get(i, {}) 
        temp[j+offset] = curr_id 
        vq_to_trans[i] = temp 
        curr_id += 1

    vq_to_range[i] = (max(vq_to_trans[i].values()), min(vq_to_trans[i].values()))

vq_to_range[70] = (392,392) # END
vq_to_range[71] = (393,393) # PAD


def prepare_cat_id_to_group_id():
    cat_id_to_group_id = {}
    for group_id, _range in vq_to_range.items():
        end, start = _range
        for i in range(start, end+1):
            cat_id_to_group_id[i] = group_id
    return cat_id_to_group_id

cat_id_to_group_id = prepare_cat_id_to_group_id()

def prepare_cat_id_to_group_name():
    cat_id_to_group_name = {}
    for cat_id, group_id in cat_id_to_group_id.items():
        if group_id > 69:
            break
        group_name = codes[group_id]
        cat_id_to_group_name[cat_id] = group_name
    return cat_id_to_group_name

cat_id_to_group_name = prepare_cat_id_to_group_name()

def prepare_group_id_to_type_name():
    group_id_to_type_name = {}

    for group_id, cat_name in enumerate(codes):

        if group_id < 4:
            type_name = 'angle'
        elif group_id < 22:
            type_name = 'distance'
        elif group_id < 28:
            type_name = 'relX'
        elif group_id < 44:
            type_name = 'relY'
        elif group_id < 53:
            type_name = 'relZ'
        elif group_id < 66:
            type_name = 'relV'
        else:
            type_name = 'ground'

        group_id_to_type_name[group_id] = type_name
    
    return group_id_to_type_name

group_id_to_type_name = prepare_group_id_to_type_name()

def prepare_cat_id_to_category_name():
    cat_id_to_category_name = {}

    for cat_id, cat_name in id_to_name.items():

        group_name = cat_id_to_group_name[cat_id]
        group_id = cat_id_to_group_id[cat_id]
        type_name = group_id_to_type_name[group_id]

        catgory_name = f"{group_name} {type_name}"
        cat_id_to_category_name[cat_id] = catgory_name

    return cat_id_to_category_name

cat_id_to_category_name = prepare_cat_id_to_category_name()

def prepare_cat_id_name_to_group_full_name():
    cat_name_to_full_group_name = {}
    cat_id_to_full_group_name = {}

    for cat_id, cat_name in id_to_name.items():

        group_name = cat_id_to_group_name[cat_id]
        group_id = cat_id_to_group_id[cat_id]
        type_name = group_id_to_type_name[group_id]

        full_category = f"{group_name} {type_name}"
        cat_name_to_full_group_name[cat_name] = full_category
        cat_id_to_full_group_name[cat_id] = full_category

    return cat_name_to_full_group_name, cat_id_to_full_group_name

cat_name_to_full_group_name, cat_id_to_full_group_name = prepare_cat_id_name_to_group_full_name()


def prepare_cat_id_name_to_group_cat_name():
    cat_id_to_full_cat_name = {}

    for cat_id, cat_name in id_to_name.items():
        group_id = cat_id_to_group_id[cat_id]
        type_name = group_id_to_type_name[group_id]
        group_name = cat_id_to_group_name[cat_id]

        name = cat_name.replace(group_name, f"{group_name} {type_name}")

        full_category = name
        cat_id_to_full_cat_name[cat_id] = full_category

    return cat_id_to_full_cat_name

cat_id_to_full_cat_name = prepare_cat_id_name_to_group_cat_name()

def prepare_group_id_to_group_full_name():
    group_id_to_full_group_name = {}
    group_name_to_full_group_name = {}
    group_full_name_to_group_id = {}

    for group_id, group_name in enumerate(codes):

        if group_id < 4:
            type_name = 'angle'
        elif group_id < 22:
            type_name = 'distance'
        elif group_id < 28:
            type_name = 'relX'
        elif group_id < 44:
            type_name = 'relY'
        elif group_id < 53:
            type_name = 'relZ'
        elif group_id < 66:
            type_name = 'relV'
        else:
            type_name = 'ground'

        full_group_name = f"{group_name} {type_name}"
        group_id_to_full_group_name[group_id] = full_group_name
        group_name_to_full_group_name[group_name] = full_group_name
        group_full_name_to_group_id[full_group_name] = group_id

    return group_id_to_full_group_name, group_name_to_full_group_name, group_full_name_to_group_id

group_id_to_full_group_name, group_name_to_full_group_name, group_full_name_to_group_id = prepare_group_id_to_group_full_name()