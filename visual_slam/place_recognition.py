"""
PLACE RECOGNITION (SOLID - Single Responsibility Principle)
"""

import numpy as np
import cv2 as cv
import time
from typing import Tuple, List, Dict, Optional

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from visual_slam.data_structures import PlaceDescriptor


# =====================================================================
# PLACE RECOGNITION
# =====================================================================

class DescriptorBasedPlaceRecognition:
    """Place recognition using descriptor similarity and clustering."""

    def __init__(self, similarity_threshold: float = 0.5,  # LOWERED threshold
                 min_matches: int = 10,  # LOWERED threshold
                 use_clustering: bool = True,
                 max_places_in_memory: int = 1000,
                 debug: bool = True):
        self.similarity_threshold = similarity_threshold
        self.min_matches = min_matches
        self.use_clustering = use_clustering
        self.max_places_in_memory = max_places_in_memory
        self.debug = debug

        # Storage
        self.places: Dict[int, PlaceDescriptor] = {}
        self.place_descriptors_matrix: Optional[np.ndarray] = None
        self.place_ids_list: List[int] = []

        # Clustering for efficiency
        self.clusterer = None
        self.place_clusters: Dict[int, int] = {}

        # Nearest neighbors for fast similarity search
        self.nn_model = None

        # Debug counters
        self.total_queries = 0
        self.successful_queries = 0

        print(f"Descriptor-based place recognition initialized")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Min matches: {min_matches}")
        print(f"   Debug mode: {debug}")

    def add_place(self, place_id: int, descriptors: np.ndarray, pose: np.ndarray) -> None:
        """Add a new place to the recognition database."""
        if descriptors.size == 0:
            if self.debug:
                print(f"   DEBUG: Skipping place {place_id} - no descriptors")
            return

        place_desc = PlaceDescriptor(
            place_id=place_id,
            descriptors=descriptors.copy(),
            pose=pose.copy(),
            timestamp=time.time(),
            feature_count=len(descriptors)
        )

        self.places[place_id] = place_desc
        if self.debug:
            print(f"   DEBUG: Added place {place_id} with {len(descriptors)} descriptors")

        self._update_search_index()

        if len(self.places) > self.max_places_in_memory:
            self._cleanup_old_places()

    def query_similar_places(self, descriptors: np.ndarray,
                           top_k: int = 5) -> List[Tuple[int, float]]:
        """Query for similar places using descriptor matching."""
        self.total_queries += 1

        if descriptors.size == 0 or len(self.places) == 0:
            if self.debug:
                print(f"   DEBUG: Query failed - descriptors: {descriptors.size}, places: {len(self.places)}")
            return []

        if self.debug:
            print(f"   DEBUG: Querying {len(self.places)} places for similarities...")

        try:
            if self.nn_model is not None and self.place_descriptors_matrix is not None:
                results = self._query_with_nn(descriptors, top_k)
            else:
                results = self._query_brute_force(descriptors, top_k)

            if self.debug:
                print(f"   DEBUG: Found {len(results)} similar places")
                for place_id, similarity in results:
                    print(f"      Place {place_id}: similarity={similarity:.3f}")

            if len(results) > 0:
                self.successful_queries += 1

            return results
        except Exception as e:
            if self.debug:
                print(f"   DEBUG: Place recognition query failed: {e}")
            return []

    def _query_with_nn(self, descriptors: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Fast querying using nearest neighbors."""
        query_vector = np.mean(descriptors, axis=0).reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=min(top_k * 2, len(self.place_ids_list)))

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.place_ids_list):
                place_id = self.place_ids_list[idx]
                if place_id in self.places:
                    similarity = self._calculate_descriptor_similarity(descriptors, self.places[place_id].descriptors)
                    if similarity > self.similarity_threshold:
                        candidates.append((place_id, similarity))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _query_brute_force(self, descriptors: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Brute force similarity search."""
        candidates = []

        for place_id, place_desc in self.places.items():
            similarity = self._calculate_descriptor_similarity(descriptors, place_desc.descriptors)
            if similarity > self.similarity_threshold:
                candidates.append((place_id, similarity))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _calculate_descriptor_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Calculate similarity between two descriptor sets with enhanced debugging."""
        try:
            # Convert to float32 for OpenCV
            d1 = desc1.astype(np.float32)
            d2 = desc2.astype(np.float32)

            if d1.shape[0] == 0 or d2.shape[0] == 0:
                if self.debug:
                    print(f"      DEBUG: Empty descriptors - d1: {d1.shape[0]}, d2: {d2.shape[0]}")
                return 0.0

            # Use BF matcher as fallback if FLANN fails
            try:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(d1, d2, k=2)
                matcher_type = "FLANN"
            except Exception as e:
                if self.debug:
                    print(f"      DEBUG: FLANN failed, using BF matcher: {e}")
                bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
                matches = bf.knnMatch(d1, d2, k=2)
                matcher_type = "BF"

            good_matches = 0
            total_matches = 0

            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    total_matches += 1
                    if m.distance < 0.8 * n.distance:  # Slightly relaxed ratio
                        good_matches += 1
                elif len(match_pair) == 1:
                    good_matches += 1
                    total_matches += 1

            if total_matches == 0:
                if self.debug:
                    print(f"      DEBUG: No matches found with {matcher_type}")
                return 0.0

            # Calculate similarity as ratio of good matches to descriptor count
            similarity = good_matches / max(len(desc1), len(desc2))


            if self.debug and similarity > 0.3:
                print(f"DEBUG PR: High similarity {similarity:.3f} detected in place recognition")

            return min(similarity, 1.0)

        except Exception as e:
            if self.debug:
                print(f"      DEBUG: Descriptor matching failed, using cosine fallback: {e}")

            # Fallback to cosine similarity
            try:
                mean1 = np.mean(desc1, axis=0).reshape(1, -1)
                mean2 = np.mean(desc2, axis=0).reshape(1, -1)
                cosine_sim = cosine_similarity(mean1, mean2)[0, 0]
                similarity = max(0.0, cosine_sim)

                if self.debug:
                    print(f"      DEBUG: Cosine similarity fallback: {similarity:.3f}")

                return similarity
            except Exception as e2:
                if self.debug:
                    print(f"      DEBUG: All similarity calculations failed: {e2}")
                return 0.0

    def _update_search_index(self):
        """Update the search index for fast querying."""
        try:
            if len(self.places) < 2:
                return

            descriptors_list = []
            place_ids_list = []

            for place_id, place_desc in self.places.items():
                if place_desc.descriptors.size > 0:
                    mean_desc = np.mean(place_desc.descriptors, axis=0)
                    descriptors_list.append(mean_desc)
                    place_ids_list.append(place_id)

            if len(descriptors_list) == 0:
                return

            self.place_descriptors_matrix = np.array(descriptors_list)
            self.place_ids_list = place_ids_list

            self.nn_model = NearestNeighbors(
                n_neighbors=min(10, len(descriptors_list)),
                algorithm='auto',
                metric='cosine'
            )
            self.nn_model.fit(self.place_descriptors_matrix)

            if self.use_clustering and len(descriptors_list) >= 3:
                self._update_clustering()

        except Exception as e:
            print(f"Failed to update search index: {e}")

    def _update_clustering(self):
        """Update place clustering for efficient organization."""
        try:
            if self.place_descriptors_matrix is None or len(self.place_descriptors_matrix) < 3:
                return

            clusterer = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            cluster_labels = clusterer.fit_predict(self.place_descriptors_matrix)

            for i, place_id in enumerate(self.place_ids_list):
                if i < len(cluster_labels):
                    self.place_clusters[place_id] = int(cluster_labels[i])

            self.clusterer = clusterer

        except Exception as e:
            print(f"Clustering update failed: {e}")

    def _cleanup_old_places(self):
        """Remove old places to manage memory."""
        if len(self.places) <= self.max_places_in_memory:
            return

        sorted_places = sorted(self.places.items(), key=lambda x: x[1].timestamp)
        remove_count = len(self.places) - self.max_places_in_memory + 10

        for i in range(remove_count):
            place_id, _ = sorted_places[i]
            if place_id in self.places:
                del self.places[place_id]
            if place_id in self.place_clusters:
                del self.place_clusters[place_id]

        self._update_search_index()
