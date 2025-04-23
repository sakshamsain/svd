var svd = {
    learningRate: 0.02,
    totalIterations: 0,
    rmse: 0,
    sigma: [],

    factorize: function(matrix, featWide, featTall, predictionMatrix, errorMatrix, learningIterations) {
        var regularizationTerm = 0.01;
        var features = featWide[0].length;

        for (var i = 0; i < learningIterations; i++) {
            var squaredError = 0.0;
            var count = 0;

            for (var row = 0; row < matrix.length; row++) {
                for (var col = 0; col < matrix[0].length; col++) {
                    if (matrix[row][col] !== 0) {
                        var prediction = this.dotProduct(featWide[row], featTall[col]);
                        var error = matrix[row][col] - prediction;
                        
                        predictionMatrix[row][col] = prediction;
                        errorMatrix[row][col] = error;
                        
                        squaredError += error * error;
                        count++;

                        // Update features
                        for (var feat = 0; feat < features; feat++) {
                            featWide[row][feat] += this.learningRate * (
                                error * featTall[col][feat] - 
                                regularizationTerm * featWide[row][feat]
                            );
                            
                            featTall[col][feat] += this.learningRate * (
                                error * featWide[row][feat] - 
                                regularizationTerm * featTall[col][feat]
                            );
                        }
                    } else {
                        predictionMatrix[row][col] = this.dotProduct(featWide[row], featTall[col]);
                        errorMatrix[row][col] = 0;
                    }
                }
            }

            this.totalIterations++;
            this.rmse = count > 0 ? Math.sqrt(squaredError / count) : 0;
            this.learningRate *= 0.99;
        }
        
        // Calculate singular values
        this.sigma = this.computeSigma(featWide);
    },

    dotProduct: function(left, right) {
        var prod = 0;
        for (var i = 0; i < left.length; i++) {
            prod += left[i] * right[i];
        }
        return prod;
    },

    computeSigma: function(U) {
        const features = U[0].length;
        const sigma = new Array(features).fill(0);
        for (let f = 0; f < features; f++) {
            let sum = 0;
            for (let r = 0; r < U.length; r++) {
                sum += U[r][f] ** 2;
            }
            sigma[f] = Math.sqrt(sum);
        }
        return sigma;
    }
};
