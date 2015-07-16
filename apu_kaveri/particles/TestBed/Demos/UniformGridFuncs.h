int4 ugConvertToGridCrd(const float4 pos, float gridScale)
{
	int4 g;
	g.x = floor(pos.x*gridScale);
	g.y = floor(pos.y*gridScale);
	g.z = floor(pos.z*gridScale);
	return g;
}

int ugGridCrdToGridIdx(const int4 g, int nCellX, int nCellY, int nCellZ)
{
	return g.x+g.y*nCellX+g.z*nCellX*nCellY;
//	return g.x+g.z*nCellX+g.y*nCellX*nCellZ;
}
