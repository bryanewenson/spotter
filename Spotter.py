import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from collections import defaultdict
import os

class Data:
	__feature_dtypes = {'bar_conf_avg' : float, 'bar_conf_std' : float, 'bar_dur_avg' : float, 'bar_dur_std' : float, 'beat_conf_avg' : float, 'beat_conf_std' : float, 'beat_dur_avg' : float, 'beat_dur_std' : float, 'tatum_conf_avg' : float, 'tatum_conf_std' : float, 'tatum_dur_avg' : float, 'tatum_dur_std' : float, 'bpm' : int, 'sect_conf_avg' : float, 'sect_conf_std' : float, 'sect_dur_avg' : float, 'sect_dur_std' : float, 'num_sections' : int, 'key_conf_avg' : float, 'key_conf_std' : float, 'dominant_key' : 'category', 'key_changes' : int, 'key_relfreq_0' : float, 'key_relfreq_1' : float, 'key_relfreq_2' : float, 'key_relfreq_3' : float, 'key_relfreq_4' : float, 'key_relfreq_5' : float, 'key_relfreq_6' : float, 'key_relfreq_7' : float, 'key_relfreq_8' : float, 'key_relfreq_9' : float, 'key_relfreq_10' : float, 'key_relfreq_11' : float, 'loudness_diff_avg' : float, 'loudness_diff_std' : float, 'mode_conf_avg' : float, 'mode_conf_std' : float, 'mode_changes' : int, 'mode_avg' : float, 'tempo_conf_avg' : float, 'tempo_conf_std' : float, 'tempo_avg' : float, 'tempo_std' : float, 'tempo_diff_avg' : float, 'tempo_diff_std' : float, 'timesig_conf_avg' : float, 'timesig_conf_std' : float, 'timesig_changes' : int, 'dominant_timesig' : 'category', 'timesig_relfreq_3' : float, 'timesig_relfreq_4' : float, 'timesig_relfreq_5' : float, 'timesig_relfreq_6' : float, 'timesig_relfreq_7' : float, 'segm_conf_avg' : float, 'segm_conf_std' : float, 'segm_dur_avg' : float, 'segm_dur_std' : float, 'num_segments' : int, 'pitch_sum_avg' : float, 'pitch_sum_std' : float, 'pitch_reldom_avg_0' : float, 'pitch_reldom_std_0' : float, 'pitch_reldom_avg_1' : float, 'pitch_reldom_std_1' : float, 'pitch_reldom_avg_2' : float, 'pitch_reldom_std_2' : float, 'pitch_reldom_avg_3' : float, 'pitch_reldom_std_3' : float, 'pitch_reldom_avg_4' : float, 'pitch_reldom_std_4' : float, 'pitch_reldom_avg_5' : float, 'pitch_reldom_std_5' : float, 'pitch_reldom_avg_6' : float, 'pitch_reldom_std_6' : float, 'pitch_reldom_avg_7' : float, 'pitch_reldom_std_7' : float, 'pitch_reldom_avg_8' : float, 'pitch_reldom_std_8' : float, 'pitch_reldom_avg_9' : float, 'pitch_reldom_std_9' : float, 'pitch_reldom_avg_10' : float, 'pitch_reldom_std_10' : float, 'pitch_reldom_avg_11' : float, 'pitch_reldom_std_11' : float, 'dominant_pitch' : 'category', 'timbre_coeff_avg_0' : float, 'timbre_coeff_std_0' : float, 'timbre_coeff_avg_1' : float, 'timbre_coeff_std_1' : float, 'timbre_coeff_avg_2' : float, 'timbre_coeff_std_2' : float, 'timbre_coeff_avg_3' : float, 'timbre_coeff_std_3' : float, 'timbre_coeff_avg_4' : float, 'timbre_coeff_std_4' : float, 'timbre_coeff_avg_5' : float, 'timbre_coeff_std_5' : float, 'timbre_coeff_avg_6' : float, 'timbre_coeff_std_6' : float, 'timbre_coeff_avg_7' : float, 'timbre_coeff_std_7' : float, 'timbre_coeff_avg_8' : float, 'timbre_coeff_std_8' : float, 'timbre_coeff_avg_9' : float, 'timbre_coeff_std_9' : float, 'timbre_coeff_avg_10' : float, 'timbre_coeff_std_10' : float, 'timbre_coeff_avg_11' : float, 'timbre_coeff_std_11' : float, 'timbre_normdiff_avg' : float, 'timbre_normdiff_std' : float, 'timbre_norm_avg' : float, 'timbre_norm_std' : float, 'key' : 'category', 'key_conf' : float, 'tempo' : float, 'tempo_conf' : float, 'time_signature' : 'category', 'time_signature_conf' : float, 'mode' : 'category', 'mode_conf' : float, 'fadeout_len' : float, 'fadein_len' : float, 'acousticness'  : float, 'danceability' : float,  'duration' : float,  'energy'  : float, 'instrumentalness' : float,  'liveness' : float, 'loudness' : float, 'speechiness' : float, 'valence' : float}

	__features_fname = 'features.csv'

	#TODO add functionality to append to current data using load function.
	#TODO find more places where exceptions need to be handled.
	#TODO add a function to update a playlist

	def __init__(self, data_dir = None):
		
		self.dir = data_dir

		self.F = pd.DataFrame(columns=self.get_column_labels())
		self.F = self.F.astype(self.__feature_dtypes)
		self.F['label'] = self.F['label'].astype(int)

		self.P = {}

		if data_dir is not None:
			load_response = self.load()
			if load_response is None:
				print('Data loading unsuccessful, use add_playlist() to get new data.')

	def playlist_count(self):
		return self.F['label'].nunique()

	def track_count(self):
		return len(self.F)

	def get_column_labels(self):
		try:
			return self.F.columns.tolist()
		except AttributeError:
			return ['label'] + list(self.__feature_dtypes.keys())		

	def new_class_label(self):
		if self.empty():
			return 1
		else:
			return max(self.F['label']) + 1

	def get_tracklist(self, Spot, pid):
		
		tracklist = []
		offset = 0
		
		while True:
			try:
				resp = Spot.playlist_items(pid, offset=offset, fields='items.track.id')
			except spotipy.exceptions.SpotifyException:
				print('Error reading tracklist: ' + pid + '...')
				return None

			# Exit the loop if there are no more tracks to read
			if len(resp['items']) == 0:
				break

			tracklist.extend([t['track']['id'] for t in resp['items']])		
			offset += len(resp['items'])

		return tracklist

	def add_playlist(self, Spot, playlist_ids):

		if isinstance(playlist_ids, str):
			playlist_ids = [playlist_ids]

		# Store the initial number of tracks in the feature set
		n_tracks_init = self.track_count()
		n_playlists_added = 0

		new_data = defaultdict(list)
		new_label = self.new_class_label()
		playlist_names = []

		for p in playlist_ids:
			playlist_names.append(str(Spot.playlist(p, fields='name')['name']))
			print('Attempting to read playlist: ' + playlist_names[-1] + '... ')

			playlist_tracks = self.get_tracklist(Spot, p)

			try:
				for t in playlist_tracks:
					process_track_data(get_track_data(Spot, t), new_label + n_playlists_added, new_data)
				
				n_playlists_added += 1
				print('Successfully read playlist: ' + playlist_names[-1] + '!')
			except TypeError:
				continue

		if n_playlists_added == 0:
			return 0,0

		self.F = self.F.append(pd.DataFrame(data=new_data), ignore_index=True)

		name_iter = iter(playlist_names)

		for c in range(new_label, new_label + n_playlists_added):
			self.update_metadata(c, next(name_iter))

		# Return the amount of new playlists and tracks added to the feature set
		return n_playlists_added, self.track_count() - n_tracks_init

	def update_metadata(self, class_label, playlist_name = None):

		try:
			metadata = self.P[class_label]
		except KeyError:
			metadata = {}

		if playlist_name is not None:
			metadata['playlist_name'] = playlist_name

		playlist_data = self.F[self.F['label'] == class_label].drop('label', axis=1)

		for c in playlist_data.columns:
			col_data = playlist_data[c]

			if pd.api.types.is_numeric_dtype(col_data):	
				metadata[c + '_mean'] = col_data.mean()
				metadata[c + '_std'] = col_data.std()
			elif not pd.api.types.is_object_dtype(col_data):
				metadata[c + '_mode'] = col_data.mode().tolist()

		self.P[class_label] = metadata

	def empty(self):
		return self.F.empty

	def load(self, s_dir = None):

		if s_dir is None:
			s_dir = self.dir
		
		try:
			path = s_dir + self.__features_fname
			self.F = pd.read_csv(path)

		except IOError:
			print('Error loading: ' + str(path))
			return False
		
		except TypeError:
			print('Invalid directory: ' + str(s_dir))
			return False
			
		return True

	def store(self, s_dir = None):
		if s_dir is None:
			s_dir = self.dir

		try:
			path = s_dir + self.__features_fname
			self.F.to_csv(path,header=self.get_column_labels())
	
		except IOError:
			print('Error storing: ' + str(path))
			return False
		
		except TypeError:
			print('Invalid directory: ' + str(s_dir))
			return False

		return True

	def set_directory(self, new_dir):
		self.dir = new_dir
		print('New storage directory: ' + new_dir)

	def get_separated_data(self):
		X = pd.DataFrame(self.F,copy=True)
		y = X.pop('label')

		return X,y

	def error_consistency(self, estimator = None, n_trials=50):
		K = 5
		X,y = self.get_separated_data()

		if pipe is None:
			pipe = Pipeline([('fs_percent', SelectPercentile(f_classif, percentile=40)),
							 ('scaler', StandardScaler()),
							 ('mlp', MLPClassifier(hidden_layer_sizes=(150,100,50)))])
		
		EC_calc = lambda ES_i, ES_j : len(ES_i.intersection(ES_j)) / len(ES_i.union(ES_j))
		skf = StratifiedKFold(n_splits = K, shuffle = True)

		error_sets = {}
		EC = []

		for t in range(n_trials):
			for train_idx, test_idx in skf.split(X, list(y)):
				estimator = pipe.fit(X.iloc[train_idx], list(y.iloc[train_idx]))
				pred = estimator.predict(X.iloc[test_idx])

				error_sets[t] = {idx for idx in range(len(pred)) if pred[idx] != y.iloc[test_idx[idx]]}

			EC.extend([EC_calc(error_sets[idx], error_sets[t]) for idx in range(t)])

		return error_sets, EC
	
	# Multi-layer Perceptrons support multilabel, which could produce interesting results. Will need to test it.
	# TODO: Adjust classifier parameters to optimize results
	def train_MLP_estimator(self, iterations=300, seed=None):
		pipe = Pipeline([('fs_percent', SelectPercentile(f_classif, percentile=40)),
						 ('scaler', StandardScaler()),
						 ('mlp', MLPClassifier(hidden_layer_sizes=(150,100,50)))])

		X,y = self.get_separated_data()
		cv_results = cross_validate(pipe,X,y, return_estimator=True)

		return cv_results

def initialize_Spotify(cid, cid_secret, redirect_uri, scopes='user-top-read playlist-read-private user-library-read'):
	Spot = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
	                                            client_secret=cid_secret,
    	                                        redirect_uri=redirect_uri,
        		                                scope=scopes))

	return Spot

def process_track_data(raw_data, class_label = None, track_data = defaultdict(list)):

	#TODO Could derive some more features based on available data

	add_value = lambda key, val : track_data[key].append(val)

	# A couple calculations that will be used a lot
	mean_std = lambda m : (np.mean(m), np.std(m))
	mean_std_count = lambda m : (np.mean(m), np.std(m), len(m))
	mean_std_diff = lambda m : (mean_std([abs(m[i] - m[i + 1]) for i in range(len(m) - 1)]))

	# Store the class label for this track.
	if class_label is not None:
		add_value('label', class_label)

	# Store some general features that are calculated by Spotify

	for label in ['acousticness', 'danceability', 'energy', 'fadein_len', 'instrumentalness', 'key', 'key_conf', 'liveness', 'loudness', 'mode', 'mode_conf', 'speechiness', 'tempo', 'tempo_conf', 'time_signature', 'time_signature_conf', 'valence']:
		add_value(label, raw_data[label])

	# Derive duration-based features

	duration = raw_data['duration']

	add_value('duration', duration)
	add_value('fadeout_len', duration - raw_data['start_of_fade_out'])

	# Derive features based on bars, beats and tatums
	for label in ['bars', 'beats', 'tatums']:

		tmp = raw_data[label]
		label = label[:-1]
		(conf_avg, conf_std) = mean_std(tmp['confidence'])
		add_value(label + '_conf_avg', conf_avg)
		add_value(label + '_conf_std', conf_std)

		(dur_avg, dur_std) = mean_std(tmp['duration'])
		add_value(label + '_dur_avg', dur_avg)
		add_value(label + '_dur_std', dur_std)


	add_value('bpm', round(60 * len(raw_data['beats']['duration']) / duration))

	# Derive features based on section data
	section_data = raw_data['sections']

	(sect_conf_avg, sect_conf_std) = mean_std(section_data['confidence'])
	(sect_dur_avg, sect_dur_std, num_sections) = mean_std_count(section_data['duration'])

	add_value('sect_conf_avg', sect_conf_avg)
	add_value('sect_conf_std', sect_conf_std)
	add_value('sect_dur_avg', sect_dur_avg)
	add_value('sect_dur_std', sect_dur_std)
	add_value('num_sections', num_sections)

	# TODO: What about a confidence-weighted duration average feature?
	# 	Could implement this for several other features as well, might be useful.

	# TODO: For all values with an associated confidence rating, drop the value 
	# 	if the confidence is zero. Actually, enforce a minimum confidence threshold 
	#	that can be set from 0 to 1.

	# Derive Key-based features
	(key_conf_avg, key_conf_std, key_count) = mean_std_count(section_data['key_confidence'])

	keys = section_data['key'] 
	key_changes = 0
	key_freq = [0] * 12 # There are 12 possible key values

	prior_key = keys[0]
	
	for key in keys[0:]:
		if key != prior_key:
			key_changes += 1
			prior_key = key
		
		key_freq[key] += 1

	key_relfreq = [key_freq[i] / float(key_count) for i in range(len(key_freq))]
	dominant_key = key_freq.index(max(key_freq))

	add_value('key_conf_avg', key_conf_avg)
	add_value('key_conf_std', key_conf_std)
	add_value('key_changes', key_changes)
	add_value('dominant_key', dominant_key)

	for i, key_rf in enumerate(key_relfreq):
		add_value('key_relfreq_' + str(i), key_rf)

	# Derive Loudness-based features
	loudness = section_data['loudness']

	(loudness_diff_avg, loudness_diff_std) = mean_std([abs(loudness[i] - loudness[i+1]) for i in range(len(loudness) - 1)])

	add_value('loudness_diff_avg', loudness_diff_avg)
	add_value('loudness_diff_std', loudness_diff_std)

	# Derive Mode-based features
	(mode_conf_avg, mode_conf_std) = mean_std(section_data['mode_confidence'])

	modes = section_data['mode']
	mode_changes = 0
	prior_mode = modes[0]
	for mode in modes[1:]:
		if mode != prior_mode:
			mode_changes += 1

	mode_avg = np.mean(modes)

	add_value('mode_conf_avg', mode_conf_avg)
	add_value('mode_conf_std', mode_conf_std)
	add_value('mode_changes', mode_changes)
	add_value('mode_avg', mode_avg)

	# Derive Tempo-based features
	(tempo_conf_avg, tempo_conf_std) = mean_std(section_data['tempo_confidence'])

	tempos = section_data['tempo']
	(tempo_avg, tempo_std, tempo_count) = mean_std_count(tempos)

	(tempo_diff_avg, tempo_diff_std) = mean_std([abs(tempos[i] - tempos[i+1]) for i in range(tempo_count - 1)])

	add_value('tempo_conf_avg', tempo_conf_avg)
	add_value('tempo_conf_std', tempo_conf_std)
	add_value('tempo_avg', tempo_avg)
	add_value('tempo_std', tempo_std)
	add_value('tempo_diff_avg', tempo_diff_avg)
	add_value('tempo_diff_std', tempo_diff_std)

	# Derive Time Signature-based features
	(timesig_conf_avg, timesig_conf_std, timesig_count) = mean_std_count(section_data['time_signature_confidence'])

	num_timesig = 5 # There are 5 possible time signature values
	timesig_changes = 0
	timesig_freq = [0] * num_timesig

	timesigs = section_data['time_signature']
	prior_timesig = timesigs[0]
		
	for timesig in timesigs[1:]:
		if timesig != prior_timesig:
			timesig_changes += 1
			prior_timesig = timesig

		timesig_freq[timesig-3] += 1

	timesig_relfreq = [timesig_freq[i] / float(timesig_count) for i in range(len(timesig_freq))]
	dominant_timesig = timesig_freq.index(max(timesig_freq)) + 3

	add_value('timesig_conf_avg', timesig_conf_avg)
	add_value('timesig_conf_std', timesig_conf_std)
	add_value('timesig_changes', timesig_changes)
	add_value('dominant_timesig', dominant_timesig)

	for i, timesig_rf in enumerate(timesig_relfreq):
		add_value('timesig_relfreq_' + str(i+3), timesig_rf)

	# Derive features based on segment data
	segment_data = raw_data['segments']

	# Derive Duration-based features
	(segm_conf_avg, segm_conf_std) = mean_std(segment_data['confidence'])
	(segm_dur_avg, segm_dur_std, num_segments) = mean_std_count(segment_data['duration'])

	add_value('segm_conf_avg', segm_conf_avg)
	add_value('segm_conf_std', segm_conf_std)
	add_value('segm_dur_avg', segm_dur_avg)
	add_value('segm_dur_std', segm_dur_std)
	add_value('num_segments', num_segments)

	# Derive Pitch-based features
	pitches = segment_data['pitches']
	(pitch_sum_avg, pitch_sum_std) = mean_std([sum(p) for p in pitches])
		
	pitches = list(map(list, zip(*pitches))) # transpose the matrix of pitch values

	max_pitch = 0

	for i,p in enumerate(pitches):
		(pitch_avg, pitch_std) = mean_std(p)
		add_value('pitch_reldom_avg_' + str(i), pitch_avg)
		add_value('pitch_reldom_std_' + str(i), pitch_std)

		if pitch_avg > max_pitch:
			dominant_pitch = i
			max_pitch = pitch_avg

	add_value('pitch_sum_avg', pitch_sum_avg)
	add_value('pitch_sum_std', pitch_sum_std)
	add_value('dominant_pitch', dominant_pitch)

	# Derive Timbre-based features
		
	timbres = segment_data['timbre']

	timbre_norms = [np.linalg.norm(t) for t in timbres]
	(timbre_normdiff_avg, timbre_normdiff_std) = mean_std_diff(timbre_norms)
	(timbre_norm_avg, timbre_norm_std) = mean_std(timbre_norms)

	add_value('timbre_normdiff_avg', timbre_normdiff_avg)
	add_value('timbre_normdiff_std', timbre_normdiff_std)
	add_value('timbre_norm_avg', timbre_norm_avg)
	add_value('timbre_norm_std', timbre_norm_std)
		
	timbres = list(map(list, zip(*timbres))) # transpose the matrix of timbre values

	for i,t in enumerate(timbres):
		(timbre_avg, timbre_std) = mean_std(t)

		add_value('timbre_coeff_avg_' + str(i), timbre_avg)
		add_value('timbre_coeff_std_' + str(i), timbre_std)

	# TODO: Could also derive features based on the change in individual 
	# 	timbre coefficients over time.

	return track_data

def get_track_data(Spot, tid):
	section_mapping = {'start': 0, 'duration': 1, 'confidence': 2, 'loudness': 3, 'tempo': 4, 'tempo_confidence': 5, 'key': 6, 'key_confidence': 7, 'mode': 8, 'mode_confidence': 9, 'time_signature': 10, 'time_signature_confidence': 11}

	segment_mapping = {'start': 0, 'duration': 1, 'confidence': 2, 'loudness_start': 3, 'loudness_max_time': 4, 'loudness_max': 5, 'loudness': 6, 'pitches': 7, 'timbre': 8}

	print('Reading Track ' + str(tid) + '...')

	raw_data = {'track_uri_ext':tid}

	query_resp = None
	failed_attempts = 0
	while query_resp is None:
		try:
			query_resp = Spot.audio_features(tid)[0]
		except:
			failed_attempts += 1

			print('Connection timed out while reading track ' + tid + '. Retrying...\n')

			if failed_attempts == 3:
				print('Problem reading track ' + tid + '. Aborting...\n')
				return None

	raw_data['acousticness'] = query_resp['acousticness']
	raw_data['danceability'] = query_resp['danceability']
	raw_data['energy'] = query_resp['energy']
	raw_data['instrumentalness'] = query_resp['instrumentalness']
	raw_data['liveness'] = query_resp['liveness']
	raw_data['loudness'] = query_resp['loudness']
	raw_data['speechiness'] = query_resp['speechiness']
	raw_data['valence'] = query_resp['valence']

	# Query audio analysis based on the track id passed to this function
	query_resp = None
	failed_attempts = 0
	while query_resp is None:
		try:
			query_resp = Spot.audio_analysis(tid)
		except:
			failed_attempts += 1

			print('Connection timed out while reading track ' + tid + '. Retrying...\n')

			if failed_attempts == 3:
				print('Problem reading track ' + tid + '. Aborting...\n')
				return None

	labels = ['bars','beats','tatums']

	for label in labels:
		tmp = list(zip(*[d.values() for d in query_resp[label]]))

		raw_data[label] = dict(duration=tmp[1], confidence=tmp[2])

	# PROCESS TRACK SECTION DATA

	section_data = {}

	tmp = list(zip(*[d.values() for d in query_resp['sections']]))

	for label in section_mapping.keys():
		section_data[label] = tmp[section_mapping[label]]

	raw_data['sections'] = section_data

	# PROCESS TRACK Segment DATA

	segment_data = {}
		
	tmp = list(zip(*[d.values() for d in query_resp['segments']]))

	for label in segment_mapping.keys():
		segment_data[label] = tmp[segment_mapping[label]]

	raw_data['segments'] = segment_data

	# Get spotify's overall track data

	tmp = query_resp['track']

	raw_data['key'] = tmp['key']
	raw_data['key_conf'] = tmp['key_confidence']
	raw_data['tempo'] = tmp['tempo']
	raw_data['tempo_conf'] = tmp['tempo_confidence']
	raw_data['time_signature'] = tmp['time_signature']
	raw_data['time_signature_conf'] = tmp['time_signature_confidence']
	raw_data['mode'] = tmp['mode']
	raw_data['mode_conf'] = tmp['mode_confidence']
	raw_data['duration'] = tmp['duration']
	raw_data['start_of_fade_out'] = tmp['start_of_fade_out']
	raw_data['fadein_len'] = tmp['end_of_fade_in']

	return raw_data

def classify_track(Spot, tid, estimator):
	track_features = process_track_data(get_track_data(Spot, tid))
	
	return estimator.predict(tid)
