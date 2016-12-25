


"""
  	left
The X coordinate of the left side of the box
  	right
The X coordinate of the right side of the box
  	top
The Y coordinate of the top edge of the box
  	bottom
The Y coordinate of the bottom edge of the box
"""


class Rectangle(object):
    left = 0
    bottom = 0
    top = 0
    right = 0

    height = 0
    width = 0

    # self._left, self._top, self._right, self._bottom
    def __init__(self, left, bottom, width, height):
        self.left = left
        self.bottom = bottom
        self.width = width
        self.height = height
        self.right()
        self.top()

    def right(self):
        """The width of the rectangle"""
        self.right = self.left + self.width

    def top(self):
        self.top = self.bottom + self.height

    def __repr__(self):
        return "left " + str(self.left) + " bottom " + str(self.bottom) + " width " + str(
            self.width) + " height " + str(self.height) + " right " + str(
            self.right) + " top " + str(self.top)

    def deepcopy(self):
        # print "passou no deepcopy"
        copy_obj = Rectangle(0, 0, 0, 0)
        copy_obj.left = self.left
        copy_obj.bottom = self.bottom
        copy_obj.height = self.height
        copy_obj.width = self.width
        copy_obj.right = self.right
        copy_obj.top = self.top
        return copy_obj

    def __deepcopy__(self):
        # print "passou no deepcopy"
        copy_obj = Rectangle(0, 0, 0, 0)
        copy_obj.left = self.left
        copy_obj.bottom = self.bottom
        copy_obj.height = self.height
        copy_obj.width = self.width
        copy_obj.right = self.right
        copy_obj.top = self.top
        return copy_obj

    def intersection(self, other):
        # if self.isdisjoint(other):
        #     return Rectangle(0, 0, 0, 0)
        left = max(self.left, other.left)
        bottom = max(self.bottom, other.bottom)
        right = min(self.right, other.right)
        top = min(self.top, other.top)
        w = right - left
        h = top - bottom
        if w > 0 and h > 0:
            return Rectangle(left, bottom, w, h)
        else:
            return Rectangle(0, 0, 0, 0)

    def union(self, other):
        left = min(self.left, other.left)
        bottom = min(self.bottom, other.bottom)
        top = max(self.top, other.top)
        right = max(self.right, other.right)
        w = right - left
        h = top - bottom
        return Rectangle(left, bottom, w, h)

    def isdisjoint(self, other):
        """Returns ``True`` if the two rectangles have no intersection."""
        return self.left > other.right or self.right < other.left or self.top > other.bottom or self.bottom < other.top

    def area(self):
        return self.width * self.height
      
    def jaccard_similarity(self, other):
	    intersection = 0
	    for x in xrange(self.left, self.right):
	        for y in xrange(self.bottom, self.top):
	            if (x >= other.left and x <= other.right and y >= other.bottom and y <= other.top):
	                intersection += 1
	    total = (self.top - self.bottom) * (self.right - self.left) + (other.top - other.bottom) * (other.right - other.left) - intersection
	    similarity = (float(intersection) / float(total))
	    return similarity

     
