var RefreshMatrices = function() {};
var rows = 5;
var cols = 5;
var features = 3;

var matrix = [
    [0,9,0,0,10],
    [3,0,0,5,0],
    [0,1,0,0,0],
    [0,5,1,0,3],
    [0,0,0,0,7]
];

var predictions = matrix.map(row => [...row]);
var error = matrix.map(row => row.map(() => 0));

// Random initialization
var latentWide = Array(rows).fill().map(() => 
    Array(features).fill().map(() => Math.random() * 0.2 + 0.8)
);
var latentTall = Array(cols).fill().map(() => 
    Array(features).fill().map(() => Math.random() * 0.2 + 0.8)
);

var $matrixActual = $("#matrixActual");
var $matrixPredict = $("#matrixPredict");
var $matrixError = $("#matrixError");
var $matrixU = $("#matrixU");
var $matrixSigma = $("#matrixSigma");
var $matrixVT = $("#matrixVT");
var $rmse = $("#rmse");
var $iterations = $("#iterations");

function CreateMatrixInputs($container, rows, cols) {
    for (var r = 0; r < rows; r++) {
        var $row = $("<div>");
        for (var c = 0; c < cols; c++) {
            var $cell = $("<input>")
                .addClass("m-cell")
                .attr("type", "number")
                .val(matrix[r][c]);
            $row.append($cell);
        }
        $container.append($row);
    }
}

function CreateMatrixElements($container, rows, cols) {
    for (var r = 0; r < rows; r++) {
        var $row = $("<div>");
        for (var c = 0; c < cols; c++) {
            var $cell = $("<div>").addClass("m-cell");
            $row.append($cell);
        }
        $container.append($row);
    }
}

function UpdateMatrixValue($matrix, row, col, value) {
    $matrix.children().eq(row).children().eq(col)
        .text(typeof value === 'number' ? value.toFixed(2) : '')
        .toggleClass("m-empty", value === 0);
}

$(document).ready(function() {
    CreateMatrixInputs($matrixActual, rows, cols);
    CreateMatrixElements($matrixPredict, rows + features, cols + features);
    CreateMatrixElements($matrixError, rows, cols);
    CreateMatrixElements($matrixU, rows, features);
    CreateMatrixElements($matrixSigma, features, features);
    CreateMatrixElements($matrixVT, features, cols);

    RefreshMatrices = function() {
        // Update matrices
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                UpdateMatrixValue($matrixActual, r, c, matrix[r][c]);
                UpdateMatrixValue($matrixPredict, r, c, predictions[r][c]);
                UpdateMatrixValue($matrixError, r, c, error[r][c]);
                
                // Update cell classes
                $("#matrixPredict").children().eq(r).children().eq(c)
                    .toggleClass("m-predict", matrix[r][c] === 0);
                $("#matrixError").children().eq(r).children().eq(c)
                    .toggleClass("m-error", error[r][c] !== 0);
            }
        }

        // Update latent matrices
        for (var r = rows; r < rows + features; r++) {
            for (var c = 0; c < cols; c++) {
                UpdateMatrixValue($matrixPredict, r, c, latentWide[r - rows][c]);
            }
        }
        for (var r = 0; r < rows; r++) {
            for (var c = cols; c < cols + features; c++) {
                UpdateMatrixValue($matrixPredict, r, c, latentTall[c - cols][r]);
            }
        }

        // Update SVD matrices
        for (let r = 0; r < rows; r++) {
            for (let f = 0; f < features; f++) {
                UpdateMatrixValue($matrixU, r, f, latentWide[r][f]);
            }
        }
        for (let r = 0; r < features; r++) {
            for (let c = 0; c < features; c++) {
                const value = r === c ? svd.sigma[r] || 0 : 0;
                UpdateMatrixValue($matrixSigma, r, c, value);
            }
        }
        for (let r = 0; r < features; r++) {
            for (let c = 0; c < cols; c++) {
                UpdateMatrixValue($matrixVT, r, c, latentTall[c][r]);
            }
        }
    };

    RefreshMatrices();
});

$("#btnGo").click(function() {
    $("#btnGo").hide();
    
    // Reset training state
    latentWide = Array(rows).fill().map(() => 
        Array(features).fill().map(() => Math.random() * 0.2 + 0.8)
    );
    latentTall = Array(cols).fill().map(() => 
        Array(features).fill().map(() => Math.random() * 0.2 + 0.8)
    );
    svd.totalIterations = 0;
    svd.learningRate = 0.02;

    // Read matrix values
    for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
            matrix[r][c] = +$matrixActual.children().eq(r).children().eq(c).val();
        }
    }

    var maxTraining = 100;
    var loop = setInterval(function() {
        svd.factorize(matrix, latentWide, latentTall, predictions, error, 1);
        RefreshMatrices();
        $("#rmse").text(svd.rmse.toFixed(4));
        $("#iterations").text(svd.totalIterations);
        
        if (svd.totalIterations >= maxTraining) {
            clearInterval(loop);
            $("#btnGo").show();
        }
    }, 1000);
});
