package metricsctx

// Dimensions is a set of dimensions that can be assigned for a cloudwatch metric
type Dimensions map[string]string

// Clone is a convenience function to create a copy of a dimenion set
func (d Dimensions) Clone() Dimensions {
	if d == nil {
		return nil
	}

	clone := make(Dimensions)
	for k, v := range d {
		clone[k] = v
	}
	return clone
}
