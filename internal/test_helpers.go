package internal

func smallBFDataset() *BFDataset {
	type rawRef struct {
		vec [VectorDimsPad]float32
		lbl uint8
	}
	raws := []rawRef{
		{vec: [VectorDimsPad]float32{0.01, 0.0833, 0.05, 0.8261, 0.1667, -1, -1, 0.0432, 0.25, 0, 1, 0, 0.2, 0.0416, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0109, 0.1667, 0.05, 0.3913, 0.6667, 0.3007, 0.0139, 0.0154, 0.2, 0, 1, 0, 0.15, 0.0282, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0336, 0.1667, 0.05, 0.4348, 0.6667, 0.1278, 0.0008, 0.017, 0.1, 0, 1, 0, 0.2, 0.02, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0415, 0.25, 0.05, 0.7391, 1, 0.2375, 0.0121, 0.0005, 0.2, 0, 1, 0, 0.3, 0.0493, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0291, 0.0833, 0.05, 0.3913, 0.3333, 0.3028, 0.0044, 0.028, 0.1, 0, 1, 0, 0.3, 0.043, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.5796, 0.9167, 1, 0.0435, 0, 0.0056, 0.4394, 0.4598, 0.4, 1, 0, 1, 0.85, 0.0032, 0, 0}, lbl: LabelFraud},
		{vec: [VectorDimsPad]float32{0.0035, 0.1667, 0.05, 0.4783, 0.8333, 0.2264, 0.001, 0.0488, 0.05, 0, 1, 0, 0.15, 0.0231, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.9708, 1, 1, 0.1304, 0.3333, -1, -1, 0.6657, 1, 1, 0, 1, 0.75, 0.0077, 0, 0}, lbl: LabelFraud},
		{vec: [VectorDimsPad]float32{0.0092, 0.0833, 0.05, 0.6522, 1, 0.0417, 0.0116, 0.0025, 0.1, 0, 1, 0, 0.15, 0.0101, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.3536, 0.5, 1, 0.087, 0.6667, 0.0049, 0.8445, 0.8925, 0.8, 1, 0, 1, 0.85, 0.0035, 0, 0}, lbl: LabelFraud},
	}

	ds := &BFDataset{
		Labels: make([]uint8, 0, len(raws)),
	}
	for d := 0; d < BFDims; d++ {
		ds.Dims[d] = make([]int16, 0, len(raws))
	}

	for _, r := range raws {
		for d := 0; d < BFDims; d++ {
			ds.Dims[d] = append(ds.Dims[d], QuantizeFloat32ToInt16(r.vec[d]))
		}
		ds.Labels = append(ds.Labels, r.lbl)
	}
	ds.NumRefs = len(ds.Labels)
	return ds
}
