class TrackableObject:
    def __init__(self, object_id, centroid):
        self.objectID = object_id
        self.centroids = [centroid]

        self.counted = False
        self.counted_in = False
        self.counted_out = False
        self.counted_pre_in = False
        self.counted_pre_out = False

        self.start_time = 0


