
import csv
import time
import json

def update_mapping(cocoDS, read_CSV, JSON_path):
	"""
	Updates all label/parent mappings in cocoDS using CSV (csv file) as the mapping scheme and writes to new JSON file.
	:param cocoDS (annotation file opened via json.load): the dataset whose labels/parents need to be remapped.
	:param read_CSV: csv file from spreadsheet with new mapping scheme whose whose columns are Old label, Old Parent, New Label, New Parent.
	 First row in the spreadsheet contains column headers, data begins in row two.
	:param write_to_JSON: desired file name for the output JSON file generated by remapping the dataset
	"""
	
	# set up
	print('updating mapping...')
	tic = time.time()
	ds = cocoDS
	all_cats = ds['categories']
	all_anns = ds['annotations']
	map_rows = []
	cats_by_id = {}
	# make categories accessible by cateogory id
	for cat in all_cats:
		cats_by_id[cat['id']] = cat
	
	# read the CSV
	with open(read_CSV, newline='') as csvfile:
		spamreader = csv.reader(csvfile, quotechar='|')
		for row in spamreader:
			map_rows.append(row)
	
	# CSV file now read and saved by row, skpping column headers row
	map_rows = map_rows[1:]
	created_cats = {}
	removed_cat_ids = []
	changed_old_cats = []

	ann_size = len(all_anns)
	curr_ann = 1
	for ann in all_anns[:]:
		# keep track of and display mapping progress
		if curr_ann % 10000 == 0:
			progress = curr_ann / ann_size * 100
			print('   {:0.2f}% complete'.format(progress), end='\r')
		cat_id = ann['category_id']
		cat = cats_by_id[cat_id]
		OL = cat['name']
		
		#check all remapping cases and remap accordingly
		for search_row in map_rows:
			# can identify each category from dataset by its old name (OL) and match to row in CSV
			if OL == search_row[0]:
				row = search_row
				# now we have the particular row from the CSV whose old category corresponds to this annotation's category
				OP, NL, NP = cat['supercategory'], row[2], row[3]
				
				# case: delete old category and annotations that reference it
				if NL == 'NA':
					if cat_id not in removed_cat_ids:
						all_cats.remove(cat)
						removed_cat_ids.append(cat_id)
					all_anns.remove(ann)
				# case: just change label and parent names in the category but keep category id and all its annotations the same
				elif (NL, NP) not in created_cats:
					cat['name'] = NL
					cat['supercategory'] = NP
					created_cats[(NL, NP)] = cat['id']
					changed_old_cats.append(cat)
				# case: the new category already exists and we need to point all old annotations referencing the old category to the new one
				# (then delete old category)
				elif (NL, NP) in created_cats and cat not in changed_old_cats:
					new_cat_id = created_cats[(NL, NP)]
					ann['category_id'] = new_cat_id
					if cat_id not in removed_cat_ids:
						all_cats.remove(cat)
						removed_cat_ids.append(cat_id)
		curr_ann += 1

	# need to update all category ids to be sequential as well as the annotation category ids that reference them
	total_cats = len(all_cats)
	unique_cats = []
	for cat in all_cats:
		if cat not in unique_cats:
			unique_cats.append(cat)
	
	old_to_seq_id = {} # old category id - > new sequential id
	for k, cat in enumerate(unique_cats):
		old_to_seq_id[cat['id']] = k
	
	# now all category ids have a unique mapping to a new sequential category ID

	# set new ids for categories and annotations to new sequential values
	for ann in all_anns:
		cat_id = ann['category_id']
		cat = cats_by_id[cat_id]
		cat['id'], ann['category_id'] = old_to_seq_id[cat_id], old_to_seq_id[cat_id]

	# write to JSON
	print('writing to JSON...')
	with open(JSON_path, 'w') as json_file:
		json.dump(ds, json_file)

	# display clock time
	stamp = time.time() - tic
	minutes = stamp // 60
	seconds = stamp % 60
	print('Done (t={} min {:0.2f} sec)'.format(minutes, seconds))
