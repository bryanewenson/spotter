"""
This module enables the user to extract predefined features from the tracks in
their Spotify playlists. Once the feature set has been collected, this module
allows the user to classify a Spotify track into one of their playlists.

Classification accuracy is low until some multilabel optimization methods
are implemented.

TODO:
	- Provide examples on class usage.
	- Add __repr__ and __str__ for classes.
	- Elaborate in some of the method docs.
	- Add multilabel feature selection steps to estimator pipeline.
	- Implement custom Stratified KFold that accepts multilabel input.
	- Implement (or find) custom scoring method(s) for multilabel problems.
	- Implement iterative classification for multilabel.
	- Add functionality to append to current dataset using load method.
	- Extract more/better features from raw Spotify data.
		- Could do more with confidence values. (weights, drop threshold)
	- Implement clustering algorithm to create playlists from a set of tracks.
	- Visualize track and playlist metadata.

"""

from __future__ import annotations
import pickle
from collections import defaultdict
from typing import (
	List, Dict, Tuple, Optional, DefaultDict, Any, Set, Callable)
import pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
	cross_validate,
	KFold,
	GridSearchCV,
)
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
# from sklearn.metrics import accuracy_score
import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth
import config

class SpotifyDataset:
	"""
	Acts as a container for the user's relevant Spotify data.

	Attributes
	----------
	feature_set: A pandas.DataFrame instance containing processed features
		for each track in the dataset.
	playlist_metadata: A dict mapping playlist URIs(str) to their
		respective metadata.
	track_metadata: A dict mapping track URIs(str) to their
		respective metadata.
	storage_dir: A path.PathDir instance representing the directory where the
		SpotifyDataset is stored.

	"""

	__feature_dtypes = config.feature_dtypes
	__file_names = config.file_names
	_spot_helper = None

	def __init__(
			self,
			storage_dir: pathlib.Path = None,
			spot_helper: Optional[SpotifyHelper] = None):

		"""
		Args:
			storage_dir: Optional; A pathlib.Path representing the directory
				where the SpotifyDataset is or will be stored.
			spot_helper: Optional; A SpotifyHelper instance. If not provided,
				will create a SpotifyHelper with the default parameters
				within config.py.
		"""

		if not isinstance(spot_helper, SpotifyHelper):
			self._spot_helper = SpotifyHelper()

		self.storage_dir = storage_dir
		self._next_class_label = None

		self.feature_set = pd.DataFrame(columns=self.get_column_labels())
		self.feature_set = self.feature_set.astype(self.__feature_dtypes)

		# PLAYLIST METADATA (KEY= Playlist URI: str)
		# name: str,
		# class_label: int,
		# (avg AND std) OR mode for each feature in feature_set,
		self.playlist_metadata = {}

		# TRACK METADATA (KEY= Track URI: str)
		# class_labels: set[int],
		# index: int,
		# name: str,
		# artist: str,
		# album: str
		self.track_metadata = defaultdict(dict)

	@property
	def num_tracks(self):
		"""The number of unique tracks in the dataset."""
		return len(self.feature_set)

	@property
	def num_playlists(self):
		"""The number of unique playlists in the dataset."""
		return len(self.playlist_metadata)

	@property
	def storage_dir(self):
		"""The directory in which to store this dataset."""
		return self._storage_dir

	@storage_dir.setter
	def storage_dir(self, s_dir: pathlib.Path):
		if isinstance(s_dir, pathlib.Path):
			self._storage_dir = s_dir
		else:
			print(f'Error: Invalid type for storage directory ({type(s_dir)})')

	@classmethod
	def from_directory(
			cls,
			directory: pathlib.Path) -> SpotifyDataset:
		"""
		Returns a SpotifyDataset from the data stored in the given directory.

		Args:
			directory: A pathlib.Path representing the directory from which
				to load Spotify data. This directory must contain files
				features.csv, playlists.pkl, and tracks.pkl.

		"""

		new_dataset = cls(storage_dir=directory)
		if new_dataset.load():
			return new_dataset

		return None

	def load(
			self,
			s_dir: Optional[pathlib.Path] = None) -> bool:
		"""
		Load data from a given directory.

		Args:
			s_dir: Optional; A pathlib.Path representing the directory from
				which to load the data. Will load from self.storage_dir if
				s_dir is not provided. In either case, the directory must
				contain files features.csv, playlists.pkl, and tracks.pkl.

		Returns:
			True if the load was succesful, otherwise false.

		"""

		if s_dir is None:
			s_dir = self.storage_dir

		try:
			path = s_dir / self.__file_names['feature_set']
			self.feature_set = pd.read_csv(path, index_col=0)

			path = s_dir / self.__file_names['playlist_metadata']
			with open(path, 'rb') as f:
				self.playlist_metadata = pickle.load(f)

			path = s_dir / self.__file_names['track_metadata']
			with open(path, 'rb') as f:
				self.track_metadata = pickle.load(f)

		except IOError:
			print(f'Error loading: {path}')
			return False

		except TypeError:
			print(f'Invalid directory: {s_dir}')
			return False

		return True

	def store(
			self,
			s_dir: Optional[pathlib.Path] = None) -> bool:
		"""
		Store data in a given directory.

		Args:
			s_dir: Optional; A pathlib.Path representing the directory where
				the data will be stored. If not provided, will store in
				self.storage_dir. Will produce files features.csv,
				playlists.pkl, and tracks.pkl in the given directory,
				overwriting any existing files with the same names.

		Returns:
			True if the store was succesful, otherwise false.

		"""
		if not isinstance(s_dir, pathlib.Path):
			s_dir = self.storage_dir

		try:
			path = s_dir / self.__file_names['feature_set']
			self.feature_set.to_csv(path, header=self.get_column_labels())

			path = s_dir / self.__file_names['playlist_metadata']
			with open(path, 'wb+') as f:
				pickle.dump(self.playlist_metadata, f, pickle.HIGHEST_PROTOCOL)

			path = s_dir / self.__file_names['track_metadata']
			with open(path, 'wb+') as f:
				pickle.dump(self.track_metadata, f, pickle.HIGHEST_PROTOCOL)

		except IOError:
			print(f'Error storing: {path}')
			return False

		except TypeError:
			print(f'Invalid directory: {s_dir}')
			return False

		return True

	def get_column_labels(self) -> list:
		"""
		Returns the list of column titles from the feature set.

		"""

		try:
			return self.feature_set.columns.tolist()
		except AttributeError:
			return list(self.__feature_dtypes.keys())

	def _new_class_label(self) -> int:
		"""
		Returns a new unique class label.

		"""

		if len(self.playlist_metadata) == 0:
			self._next_class_label = 1
			return 0
		elif self._next_class_label is None:
			return max([p['class_label'] for p in self.playlist_metadata.values()]) + 1

		class_label = self._next_class_label
		self._next_class_label += 1

		while(self._next_class_label in [p['class_label'] for p in self.playlist_metadata.values()]):
			self._next_class_label += 1

		return class_label

	def _ids_to_labels(
			self,
			playlist_ids: List[str]
		) -> np.array:
		"""
		Returns the class labels associated with the given playlist ids.

		Args:
			playlist_ids: A list containing the Spotify URIs of the playlists
				for which the class labels are desired.

		"""

		if isinstance(playlist_ids, str):
			playlist_ids = [playlist_ids]

		try:
			class_labels = [self.playlist_metadata[pid]['class_label'] for pid in playlist_ids]
		except KeyError:
			print('Invalid playlist id. Aborting')
			return None

		return np.array(class_labels)

	def add_playlists(
			self,
			playlist_ids: List[str]
		) -> Tuple[List[str], int]:
		"""
		Acquire the Spotify data for all tracks within a number of playlists.

		Args:
			playlist_ids: A list containing the Spotify URIs of playlists to
				add to the SpotifyDataset. If a playlist URI has already been
				added, this method checks for any new tracks within
				the playlist.

		Returns:
			Returns a tuple containing a list of names representing the
			playlists that were updated and the number of tracks that
			were added.

		"""

		if isinstance(playlist_ids, str):
			playlist_ids = [playlist_ids]

		n_tracks_init = new_index = len(self.feature_set)

		track_data = defaultdict(list)

		updated_playlists = []

		# Process each playlist, reading data for all new tracks
		for p in playlist_ids:
			playlist_title = self._spot_helper.get_playlist_name(p)
			print(f'Attempting to read playlist: {playlist_title}...')

			tracklist = set(self._spot_helper.get_playlist_tracks(p))

			# Ensure that tracklist returned a proper list of track ids
			if not tracklist:
				print(f'Error reading tracks for playlist: {playlist_title}. Skipping...')
				continue

			try:
				class_label = self.playlist_metadata[p]['class_label']
			except KeyError:
				# Create a new class label if this playlist is new
				class_label = self._new_class_label()
				self.playlist_metadata[p] = {'class_label': class_label, 'title': playlist_title}

			updated = False

			# Process tracks for addition into the dataset
			for t in tracklist:
				try:
					self.track_metadata[t]['labels'].add(class_label)
				except KeyError:
					if self._spot_helper.process_track(t, track_data, self.track_metadata) is not None:
						self.track_metadata[t]['labels'] = {class_label}
						self.track_metadata[t]['index'] = new_index
						new_index = new_index + 1
						updated = True

			if updated:
				updated_playlists.append(p)
			print(f'Successfully read playlist: {playlist_title}!')

		# Add new tracks to dataset
		self.feature_set = self.feature_set.append(pd.DataFrame(data=track_data), ignore_index=True)

		# Update metadata to account for new tracks in playlists
		for p in updated_playlists:
			self._update_playlist_metadata(p)

		# Return the amount of updated playlists and tracks added to the feature set
		return updated_playlists, len(self.feature_set) - n_tracks_init

	def get_row_indices(
			self,
			playlist_ids: List[str]
		) -> np.array:
		"""
		Returns feature set row indices of the tracks from a set of playlists.

		Args:
			playlist_ids: A list containing the Spotify URIs of playlists for
				which to get the indices.

		Returns:
			A numpy.array containing the row indices of all the tracks
			contained within at least one of the given playlists. These
			indices refer to the rows of the feature set pandas.DataFrame.

		"""

		class_labels = self._ids_to_labels(playlist_ids)
		indices = set()

		for c in class_labels:
			indices.update([t['index'] for t in self.track_metadata.values() if c in t['labels']])

		return np.array(list(indices))

	def _update_playlist_metadata(
			self,
			playlist_id: str
		) -> bool:
		"""
		Calculate and store metadata for the given playlist.

		Args:
			playlist_id: A Spotify URI representing the playlist for which
				metadata should be updated.

		Returns:
			True if the metadata calculation was successful, otherwise false.

		"""

		try:
			new_metadata = self.playlist_metadata[playlist_id]
		except KeyError:
			print('Playlist not found. Aborting metadata update.')
			return False

		playlist_data = self.feature_set.iloc[self.get_row_indices(playlist_id)]

		for c in playlist_data.columns:
			col_data = playlist_data[c]

			if col_data.dtype in [bool, object]:
				new_metadata[f'{c}_mode'] = col_data.mode().tolist()
			else:
				new_metadata[f'{c}_mean'] = col_data.mean()
				new_metadata[f'{c}_std'] = col_data.std()

		self.playlist_metadata[playlist_id] = new_metadata
		return True

	def formatted_labels(
			self,
			playlist_ids: Optional[List[str]] = None) -> np.ndarray:
		"""
		Return formatted labels for use with sklearn multilabel classifiers.

		Args:
			playlist_ids: Optional; A list of Spotify URIs representing the
				playlists that should be included in the list of labels. If
				this is provided, any tracks that do not belong to one of the
				playlists will be omitted from the label list. If not
				provided, all the playlists will be included.

		Return:
			A numpy.ndarray in the format expected by sklearn multilabel
			classifiers. The number of columns corresponds to the number of
			playlists in the dataset or the length of playlist_ids if it was
			given. The number of rows corresponds to the number of unique
			tracks present in the dataset or in all the playlists from
			playlist_ids if given.

		"""

		labels = np.zeros([len(self.feature_set), len(self.playlist_metadata)], dtype=np.int8)
		for track in self.track_metadata.values():
			labels[track['index']][list(track['labels'])] = 1

		if playlist_ids is None:
			return labels

		indices = self.get_row_indices(playlist_ids)
		classes = self._ids_to_labels(playlist_ids)

		return labels[indices[:, None], classes]

	def get_separated_data(
			self,
			playlist_ids: Optional[List[str]] = None
		) -> Tuple[pd.DataFrame, np.ndarray]:
		"""
		Return the feature set and formatted labels of the dataset.

		Args:
			playlist_ids: Optional; A list of Spotify URIs representing
				the playlists for which to return data. If this is provided,
				any	tracks that do not belong to one of the playlists will be
				omitted from the feature set and label list. If not provided,
				the full feature set and label list will be returned.

		Returns:
			A tuple containing the feature set and label list for the given set
			of playlists.

		"""

		if playlist_ids is not None:
			indices = self.get_row_indices(playlist_ids)
			return self.feature_set.iloc[indices], self.formatted_labels(playlist_ids)

		return self.feature_set, self.formatted_labels()


class SpotifyHelper:
	"""
	Interfaces with the Spotify API and processes incoming data.

	"""
	def __init__(
			self,
			client_id: Optional[str] = None,
			client_secret: Optional[str] = None,
			redirect_uri: Optional[str] = None):
		"""
		Args:
			client_id: Optional; Client ID provided by the Spotify API. If not
				provided, will attempt to retrieve from config.py.
			client_secret: Optional; Client Secret ID provided by the Spotify
				API. If not provided, will attempt to retrieve from config.py.
			redirect_uri: Optional; Redirect URI for the Spotify
				authorization. If not provided, will attempt to retrieve from
				config.py.

		"""

		if client_id is None:
			client_id = config.client_id
		if client_secret is None:
			client_secret = config.client_secret
		if redirect_uri is None:
			redirect_uri = config.redirect_uri

		self._spot_helper = spotipy.Spotify(
			auth_manager=SpotifyOAuth(
				client_id=client_id,
				client_secret=client_secret,
				redirect_uri=redirect_uri,
				scope='user-top-read playlist-read-private user-library-read'
			)
		)

	def get_playlist_name(
			self,
			pid: str
		) -> str:
		"""
		Retrieves the name of a given playlist.
		Args:
			pid: The Spotify URI of the playlist for which the name
				is desired.

		Returns:
			The name of the given playlist.

		"""

		return str(self._spot_helper.playlist(pid, fields='name')['name'])

	def get_playlist_tracks(
			self,
			pid: str
		) -> Set[str]:
		"""
		Retrieves the set of track URIs that make up the given playlist.
		Args:
			pid: The Spotify URI of the playlist for which the tracklist
				is desired.

		Returns:
			A set containing the Spotify URI of every track in the given
				playlist. If the playlist's tracks cannot be read,
				None is returned.

		"""

		tracklist = set()
		offset = 0

		while True:
			try:
				resp = self._spot_helper.playlist_items(pid, offset=offset, fields='items.track.id')
			except SpotifyException:
				return None

			# Exit the loop if there are no more tracks to read
			if len(resp['items']) == 0:
				break

			tracklist.update([t['track']['id'] for t in resp['items']])
			offset += len(resp['items'])

		return tracklist

	def get_track_data(
			self,
			track_id: str
		) -> Dict[str, Any]:
		"""
		Retrieves the raw track data from Spotify for the given track.
		Args:
			track_id: The Spotify URI of the track for which the data is desired.

		Returns:
			A dict mapping the names and values of various measurements.

		"""

		section_mapping = {
			'start': 0, 'duration': 1, 'confidence': 2, 'loudness': 3,
			'tempo': 4, 'tempo_confidence': 5, 'key': 6,
			'key_confidence': 7, 'mode': 8, 'mode_confidence': 9,
			'time_signature': 10, 'time_signature_confidence': 11
		}

		segment_mapping = {
			'start': 0, 'duration': 1, 'confidence': 2, 'loudness_start': 3,
			'loudness_max_time': 4, 'loudness_max': 5, 'loudness': 6,
			'pitches': 7, 'timbre': 8
		}

		print('Reading Track ' + str(track_id) + '...')

		raw_data = {'track_uri_ext': track_id}

		query_resp = None
		failed_attempts = 0
		while query_resp is None:
			try:
				query_resp = self._spot_helper.track(track_id)
			except AttributeError:
				print('Error: Spotify handler has not been initialized. Aborting...\n')
				return None
			except SpotifyException:
				failed_attempts += 1

				print('Problem reading track ' + track_id + '. Retrying...\n')

				if failed_attempts == 3:
					print('Maximum number of attempts reached. Aborting...\n')
					return None

		raw_data['name'] = query_resp['name']
		raw_data['album'] = query_resp['album']['name']
		raw_data['artists'] = []
		for a in query_resp['artists']:
			raw_data['artists'].append(a['name'])

		raw_data['explicit'] = query_resp['explicit']

		query_resp = None
		failed_attempts = 0
		while query_resp is None:
			try:
				query_resp = self._spot_helper.audio_features(track_id)[0]
			except SpotifyException:
				failed_attempts += 1

				print('Problem reading track ' + track_id + '. Retrying...\n')

				if failed_attempts == 3:
					print('Maximum number of attempts reached. Aborting...\n')
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
				query_resp = self._spot_helper.audio_analysis(track_id)
			except SpotifyException:
				failed_attempts += 1

				print('Connection timed out while reading track ' + track_id + '. Retrying...\n')

				if failed_attempts == 3:
					print('Problem reading track ' + track_id + '. Aborting...\n')
					return None

		labels = ['bars', 'beats', 'tatums']

		for label in labels:
			tmp = list(zip(*[d.values() for d in query_resp[label]]))

			raw_data[label] = dict(duration=tmp[1], confidence=tmp[2])

		# PROCESS TRACK SECTION DATA

		section_data = {}

		tmp = list(zip(*[d.values() for d in query_resp['sections']]))

		for label in section_mapping:
			section_data[label] = tmp[section_mapping[label]]

		raw_data['sections'] = section_data

		# PROCESS TRACK Segment DATA

		segment_data = {}

		tmp = list(zip(*[d.values() for d in query_resp['segments']]))

		for label in segment_mapping:
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

	def process_track(
			self,
			track_id: str,
			track_data: DefaultDict[List] = None,
			track_metadata: DefaultDict[Dict] = None
		) -> Tuple(DefaultDict[List], DefaultDict[Dict]):
		"""
		Retrieves and stores the features and metadata of a given track.

		Args:
			track_id: The Spotify URI of the track for which the data is desired.
			track_data: Optional; If provided, this track's features will be
				appended to this defaultdict. A list of keys that this method
				affects are found in config.py.
			track_metadata: Optional; If provided, the metadata for this track
				will be added to this defaultdict. The key for this new data
				will the given track URI.

		Returns:
			A tuple containing the processed features and the metadata of the
			given track. These will be returned regardless of the value passed
			for track_data and track_metadata.

		"""

		if track_data is None:
			track_data = defaultdict(list)
		if track_metadata is None:
			track_metadata = defaultdict(dict)

		raw_data = self.get_track_data(track_id)
		if raw_data is None:
			return None

		# add_value = lambda key, val: track_data[key].append(val)
		def add_value(key, val):
			track_data[key].append(val)

		# Some calculations that will be used a lot
		def mean_std(m):
			return np.mean(m), np.std(m)

		def mean_std_count(m):
			return np.mean(m), np.std(m), len(m)

		def mean_std_diff(m):
			return mean_std([abs(m[i] - m[i + 1]) for i in range(len(m) - 1)])

		# Store this track's metadata

		track_metadata[track_id]['name'] = raw_data['name']
		track_metadata[track_id]['album'] = raw_data['album']
		track_metadata[track_id]['artists'] = raw_data['artists']

		# Store some general features that are calculated by Spotify
		for label in [
				'acousticness', 'danceability', 'energy', 'fadein_len',
				'instrumentalness', 'key', 'key_conf', 'liveness', 'loudness',
				'mode', 'mode_conf', 'speechiness', 'tempo', 'tempo_conf',
				'time_signature', 'time_signature_conf', 'valence', 'explicit'
			]:
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

		# Derive Key-based features
		(key_conf_avg, key_conf_std, key_count) = mean_std_count(
			section_data['key_confidence'])

		# There are 12 possible values for key in the Spotify API
		key_freq = [0] * 12
		keys = section_data['key']
		key_changes = 0

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

		if num_sections < 2:
			(loudness_diff_avg, loudness_diff_std) = (0, 0)
		else:
			(loudness_diff_avg, loudness_diff_std) = mean_std(
				[abs(loudness[i] - loudness[i + 1]) for i in range(len(loudness) - 1)])

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

		if num_sections < 2:
			(tempo_diff_avg, tempo_diff_std) = (0, 0)
		else:
			(tempo_diff_avg, tempo_diff_std) = mean_std(
				[abs(tempos[i] - tempos[i + 1]) for i in range(tempo_count - 1)])

		add_value('tempo_conf_avg', tempo_conf_avg)
		add_value('tempo_conf_std', tempo_conf_std)
		add_value('tempo_avg', tempo_avg)
		add_value('tempo_std', tempo_std)
		add_value('tempo_diff_avg', tempo_diff_avg)
		add_value('tempo_diff_std', tempo_diff_std)

		# Derive Time Signature-based features
		(timesig_conf_avg, timesig_conf_std, timesig_count) = mean_std_count(
			section_data['time_signature_confidence'])

		# There are 5 potential values for time signature in the Spotify API
		num_timesig = 5
		timesig_changes = 0
		timesig_freq = [0] * num_timesig

		timesigs = section_data['time_signature']
		prior_timesig = timesigs[0]

		for timesig in timesigs[1:]:
			if timesig != prior_timesig:
				timesig_changes += 1
				prior_timesig = timesig

			timesig_freq[timesig - 3] += 1

		timesig_relfreq = [
			timesig_freq[i] / float(timesig_count)
			for i in range(len(timesig_freq))
		]
		dominant_timesig = timesig_freq.index(max(timesig_freq)) + 3

		add_value('timesig_conf_avg', timesig_conf_avg)
		add_value('timesig_conf_std', timesig_conf_std)
		add_value('timesig_changes', timesig_changes)
		add_value('dominant_timesig', dominant_timesig)

		for i, timesig_rf in enumerate(timesig_relfreq):
			add_value('timesig_relfreq_' + str(i + 3), timesig_rf)

		# Derive features based on segment data
		segment_data = raw_data['segments']

		# Derive Duration-based features
		(segm_conf_avg, segm_conf_std) = mean_std(segment_data['confidence'])
		(segm_dur_avg, segm_dur_std, num_segments) = mean_std_count(
			segment_data['duration'])

		add_value('segm_conf_avg', segm_conf_avg)
		add_value('segm_conf_std', segm_conf_std)
		add_value('segm_dur_avg', segm_dur_avg)
		add_value('segm_dur_std', segm_dur_std)
		add_value('num_segments', num_segments)

		# Derive Pitch-based features
		pitches = segment_data['pitches']
		(pitch_sum_avg, pitch_sum_std) = mean_std([sum(p) for p in pitches])

		# Transpose the matrix of pitch values.
		pitches = list(map(list, zip(*pitches)))

		max_pitch = 0

		for i, p in enumerate(pitches):
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

		# Transposes the matrix of timbre values
		timbres = list(map(list, zip(*timbres)))

		for i, t in enumerate(timbres):
			(timbre_avg, timbre_std) = mean_std(t)

			add_value('timbre_coeff_avg_' + str(i), timbre_avg)
			add_value('timbre_coeff_std_' + str(i), timbre_std)

		return track_data, track_metadata


class Classifier:
	"""
	Classifies tracks based on data from a given SpotifyDataset.

	"""

	def __init__(
			self,
			spot_data: SpotifyDataset,
			spot_helper: Optional[SpotifyHelper] = None,
			estimator: Optional[Pipeline] = None
		):
		"""
		Args:
			spot_data: A populated SpotifyDataset instance. The classifier
			will be trained using this data.
			spot_helper: Optional; A SpotifyHelper instance. If not provided,
				will create an instance using the default arguments found
				in config.py.
			estimator: Optional; A sklearn.pipeline.Pipeline instance that may
				replace the default pipeline.
		"""

		self._spot_data = spot_data

		self._spot_helper = SpotifyHelper() if spot_helper is None else spot_helper

		self._estimator = estimator
		if self._estimator is None:
			self._estimator = Pipeline([
					('scaler', StandardScaler()),
					('mlp', MLPClassifier(hidden_layer_sizes=(300, 175, 50), max_iter=2000))
				])

	def error_consistency(
			self,
			n_trials: Optional[int] = 50
		) -> Tuple[List[Set[int]], List[List[int]]]:
		"""
		Calculates the error consistency of the classifier

		Args:
			n_trials: Optional; The number of iterations over which to
				calculate the error consistency. A larger number of trials
				provides a more reliable measurement.

		Returns:
			A tuple containing a list of error sets and a matrix of error
			consistency values.

			ELABORATE FURTHER

		"""

		K = 5
		X, y = self._spot_data.get_separated_data()

		def EC_calc(ES_i, ES_j):
			return len(ES_i.intersection(ES_j)) / len(ES_i.union(ES_j))

		kfold = KFold(n_splits=K, shuffle=True)

		error_sets = []
		EC = []

		for t in range(n_trials):
			for train_idx, test_idx in kfold.split(X, y):
				tmp = self._estimator.fit(X.iloc[train_idx], y[train_idx])
				pred = tmp.predict(X.iloc[test_idx])

				error_sets.append({idx for idx in range(len(pred)) if pred[idx] not in y[test_idx[idx]]})

			EC.extend([EC_calc(error_sets[idx], error_sets[t]) for idx in range(t)])

		return error_sets, EC

	def optimize_estimator(
			self,
			parameters: Optional[Dict],
			eval_metric: Optional[Callable] = None
		) -> Tuple[int,Dict[str,Any]]:
		"""
		Sets the classifier to the optimal combination of parameters

		Args:
			parameters: Optional; A dict mapping the names of various
				classifier settings to the values that should be evaluated.
				Every permutation of these settings will be evaluated and
				have their scores compared. If not provided, a default set
				of parameters will be used.
			eval_metric: Optional; The scoring function that should be used
				to compare the results achieved by each set of parameters.

		Returns:
			A tuple containing the achieved score and parameters of the
			optimal classifier based on the Spotify Data.

		"""

		if parameters is None:
			parameters = {
				'mlp__activation': ['logistic', 'relu', 'tanh'],
				'mlp__solver': ['lbfgs', 'sgd', 'adam']
			}

		if eval_metric is None:
			eval_metric = accuracy_score

		X, y = self._spot_data.get_separated_data()

		clf = GridSearchCV(self._estimator, parameters, scoring=eval_metric, cv=5)
		clf.fit(X, y)

		self._estimator.set_params(**clf.best_params_)

		return clf.best_score_, clf.best_params_

	def cross_validate(self) -> float:
		"""
		Cross-validates the classifier.

		Returns:
			The average test score of K-Fold cross-validation (K=5).

		"""

		X, y = self._spot_data.get_separated_data()
		cv_results = cross_validate(self._estimator, X, y)

		return cv_results['test_score']

	def classify_track(
			self,
			track_id: str
		) -> List[float]:
		"""
		Determines in which playlist the given track is most likely to belong.

		Args:
			track_id: The Spotify URI of the track to classify.

		Returns:
			A list of values which represent the probability of the given
			track belonging to each playlist.

		"""
		self._estimator.fit(self._spot_data.get_separated_data())

		track_features = self._spot_helper.process_track(track_id)
		return self._estimator.predict_proba(track_features)
