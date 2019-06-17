

class Label:
    def __init__(self,
                 frame,
                 track_id,
                 type,
                 truncated,
                 occluded,
                 alpha,
                 bbox,
                 dimensions,
                 location,
                 rotation_y):
       self.frame = int(frame)
       self.track_id = int(track_id)
       self.type = str(type),
       self.truncated = int(truncated)
       self.occluded = int(occluded)
       self.alpha = float(alpha)
       self.bbox = [float(x) for x in bbox]
       self.dimensions = [float(x) for x in dimensions]
       self.location = [float(x) for x in location]
       self.rotation_y = float(rotation_y)

    def __repr__(self):
        return '<Measurement\t| Frame: {}, \tType: {} \t, TID: {}, \tLocation: {}>'.format(self.frame, self.type, self.track_id, self.location)


class IMU:
    def __init__(self, frame, vf, vl, ru):
        self.frame = frame
        self.vf = vf    # Velocity forward
        self.vl = vl    # Velocity left
        self.ru = ru    # Rotation around up

    def __repr__(self):
        return '<IMU | Frame: {}, v_f = {} m/s, v_l = {} m/s, r_u = {} rad/s> \n'.format(self.frame, round(self.vf,2), round(self.vl,2), round(self.ru,2))