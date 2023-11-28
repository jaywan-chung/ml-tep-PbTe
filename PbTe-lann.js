// Construct Problem-specific Aritifical Neural Networks.
// This is for prediction of thermoelectric properties of BiTe-based materials.
// Programmed by Dr. Jaywan Chung
// v0.2a updated on Sep 17, 2023

"use strict";

const jcApp = {
    chartHeight: 500,
    chartWidth: 500,
    minTemp: 250,  // [K]
    maxTemp: 750,
    nTempNodes: 100,
    dataLegend: 'Expt',
    plot1Legend: 'Pred 1',
    plot2Legend: 'Pred 2',
    colorRawData: '#594D5B',
    colorPlot1: '#808080',  // gray
    colorPlot2: '#1976D2',  // blue
};

class PbTeLann {
    constructor(embeddingNet, meanDictionaryNet, stdDictionaryNet) {
        this.meanLann = new LatentSpaceNeuralNetwork(embeddingNet, meanDictionaryNet);
        this.stdLann = new LatentSpaceNeuralNetwork(embeddingNet, stdDictionaryNet);
        this.meanOutputMatrix = null;
        this.stdOutputMatrix = null;
    }
    evaluate(inputMatrix) {
        const scaledInput = PbTeLann.getScaledInput(inputMatrix);
        this.meanLann.evaluate(scaledInput);
        this.stdLann.evaluate(scaledInput);
        this.meanOutputMatrix = this.meanLann.outputMatrix;
        this.stdOutputMatrix = this.stdLann.outputMatrix;
        this.scaleOutput();
    }
    static getScaledInput(inputMatrix) {
        const scaledInput = inputMatrix.clone();
        scaledInput.array[0] /= 0.1;  // scale Na [1]
        scaledInput.array[1] /= 0.1;  // scale Ag [1]
        scaledInput.array[2] /= 1000.0;  // scale temperature

        return scaledInput;
    }
    scaleOutput() {
        const y0 = this.meanOutputMatrix.getElement(0, 0);
        const y2 = this.meanOutputMatrix.getElement(2, 0);
        this.meanOutputMatrix.array[0] = Math.log(Math.exp(y0) + 1) * 5e-05;  // softplus activation
        this.meanOutputMatrix.array[1] *= 1e-04;
        this.meanOutputMatrix.array[2] = Math.log(Math.exp(y2) + 1);
        this.stdOutputMatrix.array[0] *= 5e-05;
        this.stdOutputMatrix.array[1] *= 1e-04;
    }
}

jcApp.startApp = function() {
    console.log("Starting App...");
    jcApp.initSelectRawdata();
    jcApp.initLann();

    jcApp.tempArray = jcApp.getLinearSpace(jcApp.minTemp, jcApp.maxTemp, jcApp.nTempNodes);
    jcApp.plot1Input = new Matrix(3, 1);  // x, y, temp
    jcApp.plot1ElecResiArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1SeebeckArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1ThrmCondArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1ElecResiStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1SeebeckStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot1ThrmCondStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2Input = new Matrix(3, 1);
    jcApp.plot2ElecResiArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2SeebeckArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2ThrmCondArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2ElecResiStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2SeebeckStdArray = new Float64Array(jcApp.nTempNodes);
    jcApp.plot2ThrmCondStdArray = new Float64Array(jcApp.nTempNodes);
    console.log('Memory allocated.');

    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(jcApp.activateChartsAndButtons); // activate buttons when google charts is loaded.

    // adjust style for mobile
    if (window.innerWidth <= 768) {
        jcApp.chartWidth = window.innerWidth * 0.9;
        jcApp.chartHeight = window.innerWidth * 0.9;
    }
}

jcApp.initSelectRawdata = function() {
    jcApp.select = document.getElementById("select-rawdata");
    for (const key of Object.keys(jcApp.rawdata)) {
        let opt = document.createElement("option");
        opt.value = key;
        opt.innerHTML = key;
        jcApp.select.appendChild(opt);
    }
    const copyToPlota1Button = document.getElementById("copy-to-plot1-button");
    copyToPlota1Button.addEventListener("click", jcApp.onClickCopyToPlot1Button);
    // select the first data
    // jcApp.select.options[0].selected = true;
    jcApp.select.value = 'x=0.030, y=0.000';
    jcApp.onClickCopyToPlot1Button();
    console.log("'Select Data' initialized.");
}
jcApp.onClickCopyToPlot1Button = function() {
    let dataName = jcApp.select.value;
    if (!dataName) return;  // if not selected, do nothing.
    let input = jcApp.rawdata[dataName]["input"];
    document.getElementById("plot1-composition-x").value = input[0];
    document.getElementById("plot1-composition-y").value = input[1];
}
jcApp.activateChartsAndButtons = function() {
    jcApp.initTepCharts();

    document.getElementById("predict-tep").addEventListener("click", function() {
        jcApp.predict();
        jcApp.drawCharts();    
    });
}
jcApp.predict = function() {
    jcApp.clearPrediction();

    const plot1CompositionX = parseFloat(document.getElementById("plot1-composition-x").value);
    const plot1CompositionY = parseFloat(document.getElementById("plot1-composition-y").value);
    const plot2CompositionX = parseFloat(document.getElementById("plot2-composition-x").value);
    const plot2CompositionY = parseFloat(document.getElementById("plot2-composition-y").value);

    if (Number.isFinite(plot1CompositionX) && Number.isFinite(plot1CompositionY)) {
        jcApp.plot1Input.setElement(0, 0, plot1CompositionX);
        jcApp.plot1Input.setElement(1, 0, plot1CompositionY);
        for(let i=0; i<jcApp.nTempNodes; i++) {
            jcApp.plot1Input.setElement(2, 0, jcApp.tempArray[i]);
            jcApp.lann.evaluate(jcApp.plot1Input);
            jcApp.plot1ElecResiArray[i] = jcApp.lann.meanOutputMatrix.array[0];
            jcApp.plot1SeebeckArray[i] = jcApp.lann.meanOutputMatrix.array[1];
            jcApp.plot1ThrmCondArray[i] = jcApp.lann.meanOutputMatrix.array[2];
            jcApp.plot1ElecResiStdArray[i] = jcApp.lann.stdOutputMatrix.array[0];
            jcApp.plot1SeebeckStdArray[i] = jcApp.lann.stdOutputMatrix.array[1];
            jcApp.plot1ThrmCondStdArray[i] = jcApp.lann.stdOutputMatrix.array[2];
        }
    }
    if (Number.isFinite(plot2CompositionX) && Number.isFinite(plot2CompositionY)) {
        jcApp.plot2Input.setElement(0, 0, plot2CompositionX);
        jcApp.plot2Input.setElement(1, 0, plot2CompositionY);
        for(let i=0; i<jcApp.nTempNodes; i++) {
            jcApp.plot2Input.setElement(2, 0, jcApp.tempArray[i]);
            jcApp.lann.evaluate(jcApp.plot2Input);
            jcApp.plot2ElecResiArray[i] = jcApp.lann.meanOutputMatrix.array[0];
            jcApp.plot2SeebeckArray[i] = jcApp.lann.meanOutputMatrix.array[1];
            jcApp.plot2ThrmCondArray[i] = jcApp.lann.meanOutputMatrix.array[2];
            jcApp.plot2ElecResiStdArray[i] = jcApp.lann.stdOutputMatrix.array[0];
            jcApp.plot2SeebeckStdArray[i] = jcApp.lann.stdOutputMatrix.array[1];
            jcApp.plot2ThrmCondStdArray[i] = jcApp.lann.stdOutputMatrix.array[2];
        }
    }
    console.log("Prediction complete.");
}
jcApp.clearPrediction = function() {
    jcApp.plot1Input.fill(NaN);
    jcApp.plot1ElecResiArray.fill(NaN);
    jcApp.plot1SeebeckArray.fill(NaN);
    jcApp.plot1ThrmCondArray.fill(NaN);
    jcApp.plot1ElecResiStdArray.fill(NaN);
    jcApp.plot1SeebeckStdArray.fill(NaN);
    jcApp.plot1ThrmCondStdArray.fill(NaN);
    jcApp.plot2Input.fill(NaN);
    jcApp.plot2ElecResiArray.fill(NaN);
    jcApp.plot2SeebeckArray.fill(NaN);
    jcApp.plot2ThrmCondArray.fill(NaN);
    jcApp.plot2ElecResiStdArray.fill(NaN);
    jcApp.plot2SeebeckStdArray.fill(NaN);
    jcApp.plot2ThrmCondStdArray.fill(NaN);

    console.log("Prediction cleared.");
}
jcApp.checkShowOptions = function() {
    jcApp.showData = document.getElementById("show-data").checked;
    jcApp.showPlot1 = document.getElementById("show-plot1").checked;
    jcApp.showPlot1TepCi = document.getElementById("show-plot1-tep-ci").checked;
    jcApp.showPlot1zTCi = document.getElementById("show-plot1-zT-ci").checked;
    jcApp.showPlot2 = document.getElementById("show-plot2").checked;
    jcApp.showPlot2TepCi = document.getElementById("show-plot2-tep-ci").checked;
    jcApp.showPlot2zTCi = document.getElementById("show-plot2-zT-ci").checked;
}
jcApp.initTepCharts = function() {
    jcApp.chartElecResi = new google.visualization.ComboChart(document.getElementById('chart-elec-resi'));
    jcApp.chartSeebeck = new google.visualization.ComboChart(document.getElementById('chart-seebeck'));
    jcApp.chartThrmCond = new google.visualization.ComboChart(document.getElementById('chart-thrm-cond'));
    jcApp.chartElecCond = new google.visualization.ComboChart(document.getElementById('chart-elec-cond'));
    jcApp.chartPowerFactor = new google.visualization.ComboChart(document.getElementById('chart-power-factor'));
    jcApp.chartFigureOfMerit = new google.visualization.ComboChart(document.getElementById('chart-figure-of-merit'));
    console.log("Charts initialized.");
}

jcApp.drawCharts = function() {
    jcApp.checkShowOptions();
    jcApp.drawElecResiChart();
    jcApp.drawSeebeckChart();
    jcApp.drawThrmCondChart();
    jcApp.drawElecCondChart();
    jcApp.drawPowerFactorChart();
    jcApp.drawFigureOfMeritChart();
}
jcApp.drawTepChart = function(chart, yLabel, yScale, getTepData, getPlot1TepAndCi, getPlot2TepAndCi) {
    const xLabel = "Temperature (K)";

    let data = new google.visualization.DataTable();
    data.addColumn('number', xLabel);
    data.addColumn('number', jcApp.dataLegend);
    data.addColumn('number', jcApp.plot1Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn('number', jcApp.plot2Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});

    let [tempData, tepData] = getTepData();
    if(jcApp.showData && (tempData !== null) && (tepData !== null)) {
        for(let i=0; i<tempData.length; i++) {
            data.addRow([tempData[i], tepData[i]*yScale, NaN, NaN, NaN, NaN, NaN, NaN]);
        }
    }
    let plot1Tep, plot1TepMin, plot1TepMax, plot2Tep, plot2TepMin, plot2TepMax;
    for(let i=0; i<jcApp.nTempNodes; i++) {
        [plot1Tep, plot1TepMin, plot1TepMax] = getPlot1TepAndCi(i);
        [plot2Tep, plot2TepMin, plot2TepMax] = getPlot2TepAndCi(i);
        plot1Tep *= yScale;
        plot1TepMin *= yScale;
        plot1TepMax *= yScale;
        plot2Tep *= yScale;
        plot2TepMin *= yScale;
        plot2TepMax *= yScale;
        if(!jcApp.showPlot1) {
            plot1Tep = NaN;
        }
        if(!jcApp.showPlot2) {
            plot2Tep = NaN;
        }

        data.addRow([jcApp.tempArray[i], NaN, 
            plot1Tep, plot1TepMin, plot1TepMax,
            plot2Tep, plot2TepMin, plot2TepMax]);
    }

    let options = {
      seriesType: 'line',
      series: {0: {type: 'scatter'}},
      title: yLabel,
      titleTextStyle: {bold: true, fontSize: 20,},
      hAxis: {title: xLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      vAxis: {title: yLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      legend: { position: 'bottom', alignment: 'center' },
      intervals: { style: 'area' },
      colors: [jcApp.colorRawData, jcApp.colorPlot1, jcApp.colorPlot2],
      height: jcApp.chartHeight,
      width: jcApp.chartWidth,
    };
  
    chart.draw(data, options);
}
jcApp.getTepData = function(rawdataName) {
    const selectedDataName = jcApp.select.value;
    let tempData = null;
    let tepData = null;
    if (selectedDataName) {
        tempData = jcApp.rawdata[selectedDataName]["temperature [degC]"];
        tepData = jcApp.rawdata[selectedDataName][rawdataName];    
    }
    return [tempData, tepData];
}
jcApp.getElecResiData = function() {
    return jcApp.getTepData("electrical_resistivity [Ohm m]");
}
jcApp.getSeebeckData = function() {
    return jcApp.getTepData("Seebeck_coefficient [V/K]");
}
jcApp.getThrmCondData = function() {
    return jcApp.getTepData("thermal_conductivity [W/m/K]");
}
jcApp.drawElecResiChart = function() {
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let tep, std, showCi;
            if (plotNum == 1) {
                tep = jcApp.plot1ElecResiArray[i];
                std = jcApp.plot1ElecResiStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                tep = jcApp.plot2ElecResiArray[i];
                std = jcApp.plot2ElecResiStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let tepMin = tep - 1.96*std;
            let tepMax = tep + 1.96*std;
            // Do not draw CI if user wanted, or positivity is violated
            if((!showCi) || (tepMin < 0)) {
                tepMin = NaN;
                tepMax = NaN;
            }
            return [tep, tepMin, tepMax];    
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartElecResi, "Electrical resistivity (mΩ cm)", 1e5, jcApp.getElecResiData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawSeebeckChart = function() {
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let tep, std, showCi;
            if (plotNum == 1) {
                tep = jcApp.plot1SeebeckArray[i];
                std = jcApp.plot1SeebeckStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                tep = jcApp.plot2SeebeckArray[i];
                std = jcApp.plot2SeebeckStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let tepMin = tep - 1.96*std;
            let tepMax = tep + 1.96*std;
            // Do not draw CI if user wanted
            if(!showCi) {
                tepMin = NaN;
                tepMax = NaN;
            }
            return [tep, tepMin, tepMax];    
        };
        return func;
    };
    
    jcApp.drawTepChart(jcApp.chartSeebeck, "Seebeck coefficient (μV/K)", 1e6, jcApp.getSeebeckData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawThrmCondChart = function() {
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let tep, std, showCi;
            if (plotNum == 1) {
                tep = jcApp.plot1ThrmCondArray[i];
                std = jcApp.plot1ThrmCondStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                tep = jcApp.plot2ThrmCondArray[i];
                std = jcApp.plot2ThrmCondStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let tepMin = tep - 1.96*std;
            let tepMax = tep + 1.96*std;
            // Do not draw CI if user wanted or positivity is violated
            if((!showCi) || (tepMin < 0)) {
                tepMin = NaN;
                tepMax = NaN;
            }
            return [tep, tepMin, tepMax];    
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartThrmCond, "Thermal conductivity (W/m/K)", 1, jcApp.getThrmCondData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawElecCondChart = function() {
    function getElecCondData() {
        const [tempData, elecResiData] = jcApp.getElecResiData();
        const elecCondData = elecResiData.map((x) => 1/x);
        return [tempData, elecCondData];
    };
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let elecResiTep, elecResiStd, showCi;
            if (plotNum == 1) {
                elecResiTep = jcApp.plot1ElecResiArray[i];
                elecResiStd = jcApp.plot1ElecResiStdArray[i];
                showCi = jcApp.showPlot1TepCi;
            } else if (plotNum == 2) {
                elecResiTep = jcApp.plot2ElecResiArray[i];
                elecResiStd = jcApp.plot2ElecResiStdArray[i];
                showCi = jcApp.showPlot2TepCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            let elecResiMin = elecResiTep - 1.96*elecResiStd;
            let elecResiMax = elecResiTep + 1.96*elecResiStd;
            // Do not draw CI if user wanted, or positivity is violated
            if((!showCi) || (elecResiMin < 0)) {
                elecResiMin = NaN;
                elecResiMax = NaN;
            }
            return [1/elecResiTep, 1/elecResiMax, 1/elecResiMin];
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartElecCond, "Electrical conductivity (S/cm)", 1e-2, getElecCondData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawPowerFactorChart = function() {
    function getPowerFactorData() {
        const [tempData, elecResiData] = jcApp.getElecResiData();
        const [, seebeckData] = jcApp.getSeebeckData();
        const powerFactorData = seebeckData.map(function(seebeck, i) {
            return seebeck*seebeck / elecResiData[i];
        });
        return [tempData, powerFactorData];
    };
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let elecResiTep, elecResiStd, seebeckTep, seebeckStd, showCi;
            if (plotNum == 1) {
                elecResiTep = jcApp.plot1ElecResiArray[i];
                elecResiStd = jcApp.plot1ElecResiStdArray[i];
                seebeckTep = jcApp.plot1SeebeckArray[i];
                seebeckStd = jcApp.plot1SeebeckStdArray[i];        
                showCi = jcApp.showPlot1zTCi;
            } else if (plotNum == 2) {
                elecResiTep = jcApp.plot2ElecResiArray[i];
                elecResiStd = jcApp.plot2ElecResiStdArray[i];
                seebeckTep = jcApp.plot2SeebeckArray[i];
                seebeckStd = jcApp.plot2SeebeckStdArray[i];        
                showCi = jcApp.showPlot2zTCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            const elecResiMin = elecResiTep - 1.96*elecResiStd;
            const elecResiMax = elecResiTep + 1.96*elecResiStd;
            const seebeckMin = seebeckTep - 1.96*seebeckStd;
            const seebeckMax = seebeckTep + 1.96*seebeckStd;    
            const seebeckSquared = Math.pow(seebeckTep, 2);
            let seebeckSquaredMin = Math.min(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            let seebeckSquaredMax = Math.max(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            // the above min/max fails when the sign of Seebeck coefficient changes
            if ((seebeckMin < 0) && (seebeckMax > 0)) {
                seebeckSquaredMin = 0;
            }
            // Do not draw CI if user wanted
            if (!showCi) {
                seebeckSquaredMin = NaN;
                seebeckSquaredMax = NaN;
            }
            return [seebeckSquared/elecResiTep, seebeckSquaredMin/elecResiMax, seebeckSquaredMax/elecResiMin];
        };
        return func;
    };

    jcApp.drawTepChart(jcApp.chartPowerFactor, "Power factor (mW/m/K\u00B2)", 1e3, getPowerFactorData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};
jcApp.drawFigureOfMeritChart = function() {
    function getFigureOfMeritData() {
        const [tempData, elecResiData] = jcApp.getElecResiData();
        const [, seebeckData] = jcApp.getSeebeckData();
        const [, thrmCondData] = jcApp.getThrmCondData();
        const figureOfMeritData = seebeckData.map(function(seebeck, i) {
            return seebeck*seebeck / (elecResiData[i]*thrmCondData[i]) * (tempData[i] + 273.15); // absolute temperature (K)
        });
        return [tempData, figureOfMeritData];
    };
    function getFuncTepAndCiForPlot(plotNum) {
        const func = function getPlotTepAndCi(i) {
            let elecResiTep, elecResiStd, seebeckTep, seebeckStd, thrmCondTep, thrmCondStd, showCi;
            if (plotNum == 1) {
                elecResiTep = jcApp.plot1ElecResiArray[i];
                elecResiStd = jcApp.plot1ElecResiStdArray[i];
                seebeckTep = jcApp.plot1SeebeckArray[i];
                seebeckStd = jcApp.plot1SeebeckStdArray[i];
                thrmCondTep = jcApp.plot1ThrmCondArray[i];
                thrmCondStd = jcApp.plot1ThrmCondStdArray[i];    
                showCi = jcApp.showPlot1zTCi;
            } else if (plotNum == 2) {
                elecResiTep = jcApp.plot2ElecResiArray[i];
                elecResiStd = jcApp.plot2ElecResiStdArray[i];
                seebeckTep = jcApp.plot2SeebeckArray[i];
                seebeckStd = jcApp.plot2SeebeckStdArray[i];        
                thrmCondTep = jcApp.plot2ThrmCondArray[i];
                thrmCondStd = jcApp.plot2ThrmCondStdArray[i];    
                showCi = jcApp.showPlot2zTCi;
            } else {
                throw new Error("Invalid Plot Number!");
            }
            const absTemp = jcApp.tempArray[i] + 273.15;  // absolute temperature (K)
            const elecResiMin = elecResiTep - 1.96*elecResiStd;
            const elecResiMax = elecResiTep + 1.96*elecResiStd;
            const seebeckMin = seebeckTep - 1.96*seebeckStd;
            const seebeckMax = seebeckTep + 1.96*seebeckStd;    
            const thrmCondMin = thrmCondTep - 1.96*thrmCondStd;
            const thrmCondMax = thrmCondTep + 1.96*thrmCondStd;    
            const seebeckSquared = Math.pow(seebeckTep, 2);
            let seebeckSquaredMin = Math.min(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            let seebeckSquaredMax = Math.max(seebeckSquared, Math.pow(seebeckMin, 2), Math.pow(seebeckMax, 2));
            // the above min/max fails when the sign of Seebeck coefficient changes
            if ((seebeckMin < 0) && (seebeckMax > 0)) {
                seebeckSquaredMin = 0;
            }
            // Do not draw CI if user wanted
            if (!showCi) {
                seebeckSquaredMin = NaN;
                seebeckSquaredMax = NaN;
            }
            return [seebeckSquared/(elecResiTep*thrmCondTep)*absTemp,
                seebeckSquaredMin/(elecResiMax*thrmCondMax)*absTemp,
                seebeckSquaredMax/(elecResiMin*thrmCondMin)*absTemp];
        }
        return func;
    };

    jcApp.drawTepChart(jcApp.chartFigureOfMerit, "Figure of merit zT (1)", 1, getFigureOfMeritData, getFuncTepAndCiForPlot(1), getFuncTepAndCiForPlot(2));
};


jcApp.initLann = function() {
    let jsonObj = jcApp.jsonObjPbTeLann;
    let embeddingNet = new FullyConnectedNeuralNetwork(2,
        jsonObj["embeddingNet"]["weightsArray"],
        jsonObj["embeddingNet"]["biasesArray"],
        jsonObj["embeddingNet"]["activationArray"]
    );
    let meanDictionaryNet = new FullyConnectedNeuralNetwork(4,
        jsonObj["meanDictionaryNet"]["weightsArray"],
        jsonObj["meanDictionaryNet"]["biasesArray"],
        jsonObj["meanDictionaryNet"]["activationArray"]
    );
    let stdDictionaryNet = new FullyConnectedNeuralNetwork(4,
        jsonObj["stdDictionaryNet"]["weightsArray"],
        jsonObj["stdDictionaryNet"]["biasesArray"],
        jsonObj["stdDictionaryNet"]["activationArray"]
    );
    jcApp.lann = new PbTeLann(embeddingNet, meanDictionaryNet, stdDictionaryNet);
    console.log("Machine learning model initialized.");
};

jcApp.getLinearSpace = function(x0, xf, numNodes) {
    const vec = new Float64Array(numNodes);
    const dx = (xf-x0)/(numNodes-1);
    for(let i=0; i<vec.length; i++) {
        vec[i] = (x0 + dx*i);
    };
    vec[vec.length-1] = xf;

    return vec;
};

jcApp.jsonObjPbTeLann = {"embeddingNet": {"weightsArray": [[0.29461981883553534, -1.011121406577703, -0.6476382099400309, 0.3998557474665714, -0.0640953141672665, -1.5052382859312634, 0.6200225051732645, 0.5085872004419195, 0.013247183658950803, -2.6640899891730725, 0.8581794886987376, 0.49305405451959144, 0.15069356017509733, -2.8513347724734235, 0.014778094214329053, 0.45768036180310395, -0.8445791667578666, -0.04664464552851861, 0.2429841939579598, 0.7527680301339014, -0.5911935918491721, 0.6844493208663175, -0.1135507606237455, 0.9212626422375492, 0.387503106343097, 0.408797881013226, 0.016416093832029927, 0.4402822194845044, -0.37973112345630483, -0.30138170417188853, -0.9604524837057705, 0.3015943958949262, -0.3527507587564969, 0.04767568949145427, -0.2749447257143874, -0.9488120645767951, 1.2275909944194123, -0.1266573987473332, -0.3235567954866368, -2.379667084525091, 0.18931307852844578, 0.7744144714340487, -1.1189711152745405, 0.7158117084871263, 0.8437000490004953, -3.573752878970164, 0.5654401124265345, 0.6825202188979038, 0.7797135062105904, 0.37935876947799846, 0.2647239976107531, -5.071670678226573, 1.2909118572828495, -0.3281404711929351, 0.8918900336467993, -0.3085840951503109, -0.5339123288702836, 0.6333194963634655, 1.0242384988534101, -0.7833224828532381, 1.383237863541805, -0.7799524148661727, -0.9302752127630599, 0.415092562503309], [-0.12367939541283543, 0.39201214603359513, 0.07329210182020642, 0.35184890050011425, -0.26725708609213317, 0.578784794253865, 0.02944763736009846, 0.04533742012494484, -1.421656691614296, -0.003498534391120825, -0.1196874201466736, -0.08440530824068372, 0.2349877487159333, -0.029601633492262, 0.02546995687305937, -0.34426787485440014, 0.0459417806012472, -0.3693003378876222, 1.4631415715677716, -0.18661186203151786, 0.09895152839101509, 0.1045800711993929, 0.23933321910473, 0.23909133691071777, 1.53807939139774, -0.07716216327803711, 1.1623996228027489, 1.1465640342886523, -0.1251060046947702, -0.23820701854199908, 0.8326197169770236, -0.8244637670487485, 0.0825122550103326, -0.0482551468031894, 0.21274219586572649, -0.2235080661130649, -0.0741375292037725, 0.22500451763071522, -0.05067019536116042, 0.1383524797570081, 0.2872673669354968, 0.2163790965554799, 0.2257418260579714, -0.18038029158197924, 0.005085716504551956, -0.13394528626840513, -0.09671915751968788, -0.0923747979613652, -0.21862149078679433, -0.20715257041720264, -0.02294500339581012, 0.07459223510047806, -0.07440205859759258, -0.11579646179038033, -0.1283200986911632, -0.21830186758933257, 0.16238523283057446, 0.15245345780862254, 0.022174679919479627, -0.06718186640802919, 0.34527203081758495, 0.42941503873741504, -0.2642092604591833, -0.062261476295132866, -0.26455467517845577, -0.03987992927186865, -0.2527717742895228, 0.336981702340129, -0.2673405465387203, 0.3869037853706864, 0.052541742620559814, 0.0765324576310487, -0.25650988198996455, -0.10614670637916206, -0.26717156874082393, -0.21213366158521352, -0.40136610825783536, 0.08223530300809617, -0.2004518218510099, -0.19707693109522675, 0.13129392081459593, -0.33605143816434874, 0.46171031227068055, -0.3116459385186972, -0.17427117653097346, -0.2170444019113011, 0.21493545342947123, 0.30960113189692745, 0.28873871137692847, -0.3153782559909881, 0.42222725193396066, -0.07359906838841332, -0.38334683305478107, 0.28419150750113253, 0.08664689364171259, -0.30502767942875564, -0.3555408241266615, 0.36294835512893175, 0.17831947737869822, -0.25541898900279153, -0.21152826587440754, -0.1123372306979736, 0.2717709560100152, -0.11676559488220818, 0.16654828674037023, 0.23347252274482927, 0.24077158529487117, 0.37965367130793737, -0.047481294818786414, -0.4017190176433306, 0.041418764619400715, 0.8322707308144363, 0.3144993647434717, -0.21435642425038937, -0.20677756924158155, 0.3416948074061076, 0.1905377198807485, 0.5081809140902123, 0.052105323289621974, -0.17298256052465857, -0.25583010504026094, 0.11486242214071293, -0.10811324455811672, -0.2882662109177635, 0.06284300970129938, -0.737942066395508, -0.2712516543869736, 0.1784519264016037, -0.23871444421664248, -0.13033167646910063, -0.1641600712087561, -0.07364122052156102, 0.06847140558554185, 0.052789234409490994, -0.02853767215285652, -0.06348653550401959, -0.20676030594982736, 0.14380850862954894, 0.07986900664052003, 0.1967111486325214, 0.08285621459403171, 0.27087797913287215, -0.09034925278155591, -0.12614905002311028, -0.03561431388751028, -0.2948193865463542, 0.145878337126924, -0.3107675839916779, -0.1725838124407182, -0.5440011311942115, -0.49655509702388434, -0.009006543922119314, 0.37349479525818374, 0.10016479188793073, -0.20533992715886393, -0.09037932632759764, 0.019904161669503543, 0.09518860669093777, 0.07992744902686998, -0.29855992604555925, 0.14567608264205922, -0.3065254510356007, -0.15226512068314757, 0.005396937757090839, -0.012367316233309989, 0.2353648983947025, 0.28469877542540106, -0.043304319586325045, -0.11997833752309213, -0.021403119649702283, -0.4919291570194112, 0.1941360802373793, -0.05789211136874948, -0.13899728156758226, -0.09530062360800683, 0.09478223009303843, 0.10144782322287942, -0.15183678779440166, 0.1743298093763866, 0.03599314768180224, -0.2247128937740263, 0.28824866462951776, -0.07990951984536937, -0.06702656506784437, -0.21660329439740725, 0.03717483171145161, 0.003299296811912878, 0.05685190617604111, -0.014190493369123588, -0.2790018805070555, -0.2123121653991275, -0.08788315949585689, -0.044040550222871525, 0.17856492416315936, -0.035878574351424636, 0.1358140510444889, 0.09524614592948893, -0.049074794455252715, 0.2650326249273434, 0.1827402490892451, -0.12034833885816698, 0.11606030234435472, 0.1737164177808644, 0.008568400568429883, -0.1114839683543269, 0.2058980091522467, 0.243028228564296, -0.31263765130022325, -0.04175549790959028, 0.08325896147272684, -0.11942796425880843, 0.0071168452765739115, -0.13806431555271087, 0.16886538860011754, -0.3377491701043876, -0.03330760319471108, -0.09205550048870177, 0.0129405560345173, -0.07265204033694328, 0.27637819488769805, 0.037210044883744166, 0.13093371566974812, 0.16383247517365487, -0.05625374789199033, -0.6415442603073774, 0.12111069074403615, -0.30060846978074224, -0.07395678350397192, -0.33697555884262304, -0.04455953593416881, -0.1319965383365215, 0.40675598532727186, -0.28618251844736503, -0.2638479004544357, 0.3845912002712623, 0.2624237190576574, 0.31077218228512893, 0.10392045352632945, 0.06531260894503303, 0.7581294868502566, -0.0538624424763944, -0.3612050716794022, -0.3270361657418216, -0.1312612471246952, 0.03541316199928735, 0.3589001724583287, -0.30062215236446693, -0.03435461922760261, 0.0801892866174024, 0.12379282344985094, -0.5460635470423445, -0.11646543578479336, 0.29406052105075253, -0.7388908798834367, -0.3767663431418639, 0.22702237849271337, -0.4329348073364008, -0.22760829468605698, 0.1367522816970687, -0.2780684916519585, -0.03921211713647703, -0.29042271013629795, 0.1797236029913433, 0.2796630284431298, 0.09545934673880117, 0.06714604832864156, 0.09395905906907459, -0.2562535874917554, 0.015835783516302056, 0.1124499217861902, -0.10157829359432125, -0.5506828771347974, 0.13667314644495754, 0.08400384269602555, -0.1935137157163556, 0.0011343886240386797, 0.17163396436174833, -0.4098390583393285, -0.14050071091441513, -0.019869390350154007, 0.3140264465274106, -0.12402547517348272, -0.17326156421856417, 0.07233665699273346, 0.08733520380385484, 0.29632602932188157, -0.28217708379530043, -0.32136404168889354, -0.17024164763086327, 0.2642119636871356, 0.4719531826070902, -0.4337538436624042, 0.4071787644686852, -0.2276150813448678, 0.06918213927175884, -0.23635195419781538, 0.09673556758695478, -0.5316960200036384, -0.22987061531824254, -0.20622503633184083, -0.3568308867386803, -0.05873234295541293, 0.38911959351550657, 0.5325214668416243, 0.29766518717205637, 0.5928417482665541, -0.26450136239063293, 0.5046330404542366, -0.09220329029986755, 0.053544446964482156, 0.13117139157849148, -0.025006821430711795, -0.4666659958838618, 0.1630115537505174, -0.3085150049544395, -0.20688132184395902, -0.2737515245492434, -0.010986309330774394, -0.08059769705140701, 0.016217331207950974, 0.08869981491836108, -0.1646282269114564, 0.10815638736916346, -0.016230582450310994, -0.26039895513736405, -0.009732693374234491, -0.06794251064686138, 0.01483364083286404, -0.35916023110404305, 0.15743908191530198, -0.30093095645402257, -0.17093540345461666, -0.09918844144846792, -0.10218655349784597, -0.1278367349724821, 0.4792559909282136, 0.12810442652487725, 0.2497086136116115, 0.08276402721657822, 0.31753353238299115, -0.07072356352665408, 0.2751958201685649, -0.285194750909427, 0.008111181377805572, 0.3994390317821416, -0.2403019081510919, 0.29216908939570996, 0.26405442392856365, 0.04678510895113288, 0.05736783863803785, -0.046534310442513833, -0.3918578076808742, -0.013780628798721276, 0.15729432546838534, 0.5780159910737194, -1.0890523571522768, 0.6346408599167742, -0.9442709355771163, 0.09858206837432967, -0.09280934261316756, 0.36824001119617344, -0.9731583079061049, 0.10192048942338947, -0.45832455548256884, -0.7332590180044382, -0.4128593790642671, 0.47108212124730575, 1.1885739997831546, 0.5202952479977574, 0.6511560953789854, -0.3983832864533379, 0.8483792767103033, -0.9495577536965591, 0.4527834634049585, -0.15618541920806378, -1.256366895179599, -0.4993096635959408, 0.07681797703477011, -0.5919772779921786, -0.6113028473588296, -0.12914347516433708, -0.6585843701763897, -0.47975492945637366, 0.5412177024256466, 0.0370476921238658, 0.19675133324492125, 0.2858450040284431, 0.22559815254633997, 0.0018565774146825555, 0.027837460234041714, 0.1406701061939263, -0.03315597278126418, 0.1903424162650418, 0.14408577162225766, 0.5327168443314578, 0.20436119796313293, 0.0330550876518967, 0.36519137897364606, 0.04325976308188274, 0.19142643706649184, 0.20630883310186157, -0.0563628208592349, -0.37764332910809556, -0.3446986483049938, 0.4257422700054797, 0.019850191480133575, -0.1513608681929994, 0.04995213322996325, 0.12378982896539735, -0.010434487155294473, -0.3847779515237633, 0.046301459012952496, 0.1946369821492052, 0.05385674428402979, -0.11681822109492682, -0.05764163267334207, -0.5867011973396531, 0.21994564778460846, -0.44240208054908314, -0.7273109353847577, -0.6610572328410111, -0.7502338796721443, -0.1480355257823385, 0.3851552857876292, 0.34412602507151374, -0.35531321438770663, 0.5911711856178796, -0.11787167476978319, -0.24537013926211562, 0.3665417470371963, 0.4339092274234272, 0.44153446213716185, 0.34813488755168137, -0.21384488267124924, -0.7427130765551034, -0.5596994120230229, -0.2460980262457899, 0.16448076791765384, -0.810197937417572, -0.5332118454505839, -0.44267258729287956, -0.6253323355856014, -0.4459732182233087, -0.8402516382200456, 0.36045582100834944, -0.5313514475487777, -1.2671907518322691, 0.6223676628972925, 0.47515977641066093, 0.2623019192259595, 3.1083518655120788, -0.8252739666961082, 2.6119544620134594, -0.8932152520416305, 3.0255116991540825, -0.08056930836775592, 0.001877877283892207, -1.3923878611929164, -0.5764693821435165, -2.3471677780708555, -0.4452321211223012, -0.0023940298022208767, 1.1868134486119863, 0.34806811508918006, 0.8871029177731506, 1.2010900794514807, 0.39653565578235944, 2.3030175745945662, -1.5798810037909232, 0.28797409999246404, 2.4119505233896636, -1.1716261014144285, 0.12836297100907784, 4.045880653500507, 0.9401801050227245, 1.2467020506583568, -0.21596349171082826, -0.2343315133863379, 0.5013508443066333, -1.0958260156554875, -0.43069858972679836, -0.13784116902253854, -0.19600425271953636, 0.15814877697490995, -0.3613805754589764, 0.13648585045877318, -0.042222866200517205, 0.22043118720588226, -0.34265701502768475, -0.06951864247428531, 0.02048855442989081, -0.010307406124938795, -0.07540320244258611, -0.1217634316947607, -0.19377119871329307, -0.2846739131350363, -0.027086060256155734, -0.40507696632556095, 0.4575834221192905, -0.24284412432749516, 0.10548398605707, -0.47900911149111997, -0.3491480205302315, 0.15591385281353706, 0.4739310646178294, 0.08373567061984243, 0.6194142764865929, 0.020877283922333394, -0.5005496247303406, 0.3301678462283065, 0.2676317384549265, -0.18387417587697574, -0.2560507248356831, 0.8946066163879729, 0.17157807364162977, 0.004762737158343141, 0.2029565375779121, -0.09449167910928996, 0.16372857168012311, 0.16696005768394387, -0.03601768969213883, 0.2885852340998446, 0.3023426694540104, -0.15555036429258673, 0.23777105122847503, 0.15589410515125512, 0.25653742465449586, 1.3010242120876023, 0.10435821509176109, 0.285725330792231, -0.694016174277324, 0.003639188623041787, 0.29146438246510914, 1.723949159145478, -0.056479544627575165, 0.2730362522852978, 0.021596188774823145, 0.3996659915586715, -0.423763341963705, -0.44576233943179217, 0.3898734517427327, -1.0709401165239576, -0.7068584898385705, 0.42053036263291194, -0.7755837860466326, 0.8330292334683601, 0.3167324148047471, -1.6193091042648975, -0.4967249640056662, -1.285974030255054, -0.561386664641631, 0.035000354875619054, 0.7317814557928095, -0.3752976723182081, 0.9680723507448143, -0.19031926056050252, -0.16148756885129062, 0.4238336028883825, 1.3134433688124425, 2.2222126791774164, 0.9364935045906908, 0.2359271943432249, -0.8596833577442887, 0.13512292931834666, -0.11455211957481258, 1.932513437812803, -0.7786246766296456, -1.0268471689161682, -0.42114877687388586, -0.8359140749102493, -0.5679538764318676, -0.9328225717609026, 1.3842715275185715, -1.2952912648264605, -1.1017464188884305, 0.7463949257414747, -0.3303643723461621, -0.14835229961051113, -0.40890921608946934, -0.0242176892985866, -0.21838693908383233, -0.08101518011992541, -0.4603179150072215, -0.21109153554578788, 0.05483639711633297, 0.24327709578004442, -0.29423806865696545, -0.3733571007079618, -0.11325336368318797, -0.22716872224159826, -0.1641728471683468, -0.21724774737991262, 0.11541706967937156, -0.095607693205991, 0.42272594521979356, 0.05664826547330723, 0.05975555968108586, -0.33620724536507723, -0.3446444651743236, 0.26199891350194243, 0.12190215830837581, -0.06856229544462555, 0.21496127657538183, 0.26315675353718, -0.403039583819613, -0.048099022511116044, 0.09555518861009452, -0.048022939355080785, 0.3446839748672397, -0.1504140716878702, 0.30882398499051883, -0.3264817979989352, -0.17227576112579962, -0.2646716394747584, -0.1015111042846934, -0.15555152765056837, 0.2149897987820008, -0.520525629521424, -0.05442855633570723, -0.14945182315395497, -0.17020914438566517, -0.25284268426859413, 0.1431477890185706, -0.0739952581725197, 0.35283175216581625, -0.0349904268458722, -0.4302991536551231, 0.03217845701470968, -0.5071101971651075, -0.30017244598840226, -0.21703921965272538, -0.620186268161973, -0.7062772787871342, -0.028171401340722494, 0.04417254012334013, 0.02762503876783736, -0.32311424112787573, 0.32154340474238363, -0.12364698107187208, 0.24617236658710834, -1.1465937663333827, 0.08975658036806579, -0.29111433554237476, 0.16476592857247888, 0.0033095761527081866, -0.2687052943022017, -0.2155418196741359, 0.5520714231833965, 0.015352939345842932, 0.3257860636473954, 0.5478368590374357, 0.032595305251854424, 0.40803404097293866, 0.00823384956940833, -0.038712437461595445, 1.0900368284761253, -0.01005870036041276, -0.41586976515301305, 0.06869711341871675, -0.216745642155391, 0.1508247104282004, 0.9880794588562678, -0.5971106466856245, -0.31951657489258933, 0.43564427913058607, -0.2049930825743515, -0.5193552409709448, -0.0057497664123474365, 0.7115476498719351, -0.9423419208992165, -0.5516047157461145, 0.12246904197003938, 0.013530312181863171, 0.13161804789118378, 0.25959145550740353, -0.22876399683539794, 0.06193062213430888, -0.8917175978996683, 0.061562171206096444, -0.005048307753713497, 0.7226987108339109, -0.33238046271433314, 0.047959645633739424, 0.031006377240603896, 0.0691455097741813, -0.12817112499744673, 0.37369583264221434, -0.1612166609775185, 0.5871239546009009, 0.34681019789715234, -0.7988189180047686, 0.5971223644165272, -0.47984180786845665, -0.11989904321409492, 0.1416740852039609, -0.6520206135896178, -0.4414992739819004, 0.43595377351544967, -0.8753496182765825, -0.5528086749839497, 0.1486805421727959, 0.12635018273071294, -0.3032420350520835, 0.47223718405070475, -0.04024046137647455, 0.15644275328946422, -0.10206694290429305, -0.24464274919646473, 0.12138028687856388, -0.10082105444970861, 0.08286132852148761, 0.20794322941561863, 0.23695648234885788, 0.2768085268149267, 0.08714647060711235, 0.25087167869945337, 0.19362583029007968, 0.22179943985748252, 0.2567279313722002, 0.6854232815758871, -0.12805215725937302, -0.06822701132541116, -0.24243472077701467, 0.18283283137117853, 0.06295595277881777, 0.5163036151443313, 0.3712881031989611, 0.14162864643576278, 0.021543790367035832, 0.31427194762679256, -0.0403303326577099, -0.19468114931936933, 0.5259834226153721, -0.5054815541552268, -0.13467653754274092, 0.0869813871496652, 0.30707650759594035, 0.06285265063406931, 0.3707643090793391, -0.7260403725939322, 0.25281163843624294, -1.2726332169545664, 0.05241965212346297, -0.4455269096425946, 0.8822609740976408, -0.41233456723127626, -0.33700365664332865, 0.13334407515002472, -0.6312842111964089, -0.4287860289433952, 0.234524502537255, 0.30101472455794226, 0.44955650561855987, 0.9511345128096043, -1.225402285768655, 0.15338910393382446, -0.6072772523389987, 0.2768757898613217, 0.17191744510668103, -0.9355939843718502, -1.2369010063622454, -0.2662936481078879, -0.8728021577914494, -0.33000941999684663, 0.06959270898352746, 0.2730605052652458, -0.823023140889888, 0.6396122071296452, -0.22980399636261833, 0.2919258015312581, 0.06345503938711604, -0.5647663809696953, 0.19223802485512748, -0.3559179792367639, -0.05976734375281879, 0.12794359228671995, 0.06596091131470459, -0.39195181300089477, -0.03582351030170607, -0.10447646382626766, -0.08161097195768516, 0.2417986209733021, 0.5429259092169983, 0.6311032333445384, 0.4244556631884714, 0.18864728179121626, 0.040950674676731774, 0.1383817032582558, 0.049102210904247066, 0.4951005665298892, -0.3588370803653497, -0.30976204581689726, 0.05270582073901262, -0.2580112526973435, -0.5470282880857203, -0.4027182278856834, 0.06173220687489358, -0.6601080270627028, -0.5054930243063724, -0.1354240355221484, -0.063072091009426, 0.12456546360518127, 0.31083764529717056, -0.6260918409582189, -0.11641822286368204, -0.5725590710606814, -0.1017060579183765, 0.11787000211726237, 0.2533741450903571, -0.544294313436276, -0.40470752847356456, -0.0784768932676658, -0.5634577666541714, -0.24975736231421744, 0.0485909038230101, -0.3688237854131239, 0.39163658308899335, 0.45784604474900237, 0.1590697511550322, -0.02741433682505928, -0.16869609736547733, -0.5507504520779115, -0.26855444726590005, -0.4518078476220386, -0.3102048494660624, -0.03025047416918678, -0.234495715262922, -0.35819129139615524, -0.35800186677752455, 0.50746297094259, 0.02422076865228733, 0.0044992655639742194, -0.344188781350239, 0.17643919472856115, -0.5277775984635751, -0.1891036070915757, -0.4221747452775651, -0.11289738383787053, -0.3600972186160828, 0.25492927379948005, 0.6207823113930816, 0.325031025695616, 0.38659303179957144, 0.4395976063773322, 0.2430516089594946, 0.003030709198499147, -0.1310084882413894, 0.18057328689767796, 0.40811684606431403, -0.017757805328218187, -0.655085355190246, -0.43633103595727707, 0.24307162742707153, 0.48370304297733024, -0.003247615039284856, -0.21648897236848413, -0.22021726460497604, -0.3456120187198464, -0.3198375224001455, -0.5379863994822282, 0.590393186371246, -0.09945409846934178, -0.6211703180795233, 0.5684187169731637, 0.32452596446358406, -0.35012133021125846, -0.1254158852656464, -0.3752405800978917, 0.2279988058340296, -0.6724677275959268, 0.14586737573669964, -0.38321192505655, -0.05555137695301632, -0.4612639143887138, -0.15370898211264963, -0.3699650390072315, -0.43576500163805937, -0.11866076299878571, 0.07432781993985273, -0.42038163546720103, -0.0009637374972190371, 0.43696604304351705, -0.04617584418971037, -0.07940634011087072, -0.4167824925191571, -0.3890245667593877, -0.0733463892007071, -0.18810718548692476, -0.2527701016122837, -0.27408842287182283, -0.3418959656077154, -0.11667672032884925, -0.4440319832739033, 0.27823694508526947, -0.049577656586277476, -0.0007174090191660492, -0.01408905225215645, -0.044910014376426236, 0.009846033430942808, 0.02761443226413688, 0.10984128405567507, 0.07704601576871603, 0.08981071278458574, 0.28031334875902286, -0.1923365825944006, -0.28617018488988905, 0.1558301826814836, 0.11063497602665394, 0.09304676415717601, 0.08724929220490801, -0.07541322616067478, 0.06608109575089602, 0.044953255698627605, 0.3148995671520688, 0.010120776996801762, -0.03540114891457302, 0.15884601582604876, 0.03088891637504193, -0.2990562424313218, -0.07144927890186836, 0.0935199710698565, 0.09005023254815656, 0.15037968213503705, -0.15026888262250973, -0.30902381159369774, 0.12378429640356427, -0.12394435595553739, -0.045408122899906575, 0.1417465923081046, 0.26406120320675863, 0.30755580591796133, -0.27197034017965005, -0.018103116463309712, -0.4489543049136978, 0.2567274225089272, -0.05529153064571402, 0.3653831037354799, -0.28242355426931787, -0.03422176472147357, -0.511973756538892, -0.42650532899222593, 0.029024287992736788, 0.11621671710466738, 0.6109371435936494, 0.1324860241087742, 0.3018395804091879, -0.4384606699105929, 0.24955538206170494, -0.659404768232624, 0.5845848153849631, 0.189257617074536, -0.3045965854531533, -0.037981645746907634, 0.12227123871344191, 0.19802895734066853, -0.0339407096346448, -0.01025554912438038, -0.27532215902496043, 0.022037541729418707, 0.10113642086176591, -0.3749541377811962, -0.3008171068501254, -0.7430559655748515, -0.3310573134780965, -0.6608756780048659, -0.5085982367959653, -0.6804880713840622, -0.1885539872788031, 0.6308276968513384, 0.21669989539199003, 0.23673413209907618, 0.28118240982120146, -0.11313169294029125, 0.25774384040890835, -0.011691238012424268, 0.28290269349355845, 0.030207661877230355, -0.2951137496759997, -0.7738661807625263, -0.20921611241836516, -0.17305619955056337, 0.1737235359682524, -1.1497409711873887, -0.02593302485588561, -0.061832106579572804, -1.1967825234135931, -1.3325390613721186, -0.9169326952319626, 0.441181279349877, -0.31688964073741427, -1.1011055771589895, 0.7302219829198839, 0.02365975940021788, 0.15247733006251818, 0.14969676005077148, 0.10845142860495191, 0.001341717340512231, 0.2175683252769845, -0.0122516495517465, 0.08816049920249282, -0.14467604856282745, -0.21785421733917995, -0.15419100809292086, 0.14280511697892817, 0.12171094624738693, -0.0929580734387389, 0.28589938099461815, 0.2989649739432929, 0.024293441856127723, -0.11563587167178574, -0.09700057706829004, -0.12751532727501563, -0.11293698587052461, -0.09846111425357525, 0.14950906008291057, -0.02345791770683315, 0.13357005852999948, -0.10751138902812399, 0.1122179944761615, -0.30186484245211787, -0.017181584239731804, 0.14810704966310015, -0.15682059914737073, -0.02093227383247038], [0.5363487864831735, 0.0002917675520955846, 0.27127183186970844, 0.15268444496751565, -0.2035835654651821, -0.003278153540242434, 0.00032711345648357183, 0.4228723712628022, -0.012633949311716566, -0.04406441602113243, -0.012625179843113638, -0.1198497592302851, 0.006620947867554927, 0.4949451385041158, -0.47469741612969857, -0.05002886565340817, 0.39518893331094007, -0.460945704869091, -0.11147780098546285, -0.2066546074103449, 0.3172016594850812, -0.17282435540396748, 0.14881624326727017, 0.4596252177680872, 0.5314211001644863, -0.12694971260976484, 0.2117156076584054, -0.15314563378064738, -0.0017886972262312728, 0.43163660216253774, -0.3506878259446736, -0.0005852345164078832, 0.4429168418136563, 0.0042269259302859866, 0.12318107763627842, -0.41837814631781606, 0.40510454945570223, 0.005476958255796144, -0.002889952477805319, -0.14532629421543586, 0.03654896754003402, -0.06445691107655488, -0.03734137573319765, -0.3491635481023771, 0.18860730981575807, 0.4144603012569605, 0.1323468165610217, 0.09226150619961063, -0.4877456211510577, -0.4106263995075028, 0.28959756618255383, -0.18115298878634944, -0.20378689260006191, 0.26809620033418574, -0.2571661877388533, 0.3878174945540388, -0.20136291209505033, 0.0328083745254505, 0.15283281861692394, -0.03515480070852109, -0.0028527972135552445, -0.2351654197981988, -0.5425681716882065, -0.0025631693921547042, -0.29226508961569636, -0.0007788912171493205, -0.07435140401701706, -0.05778177255954745, -0.07923041463256789, 0.004561895435568184, 0.0016694270872344625, 0.17246722771175427, -0.007177969653144729, -0.14083954725888462, -0.08601788212072445, -0.6088733961208684, 0.23888495989118613, 0.47226583717068815, 0.22269731815095026, -0.5381357934501997, 0.27460550291741803, -0.3112734529364373, -0.12276593640241887, 0.40804985167244373, -0.18779279021121903, 0.2257760261303448, 0.14613268708679963, -0.3529743375199442, -0.3175111115539711, 0.4349917191273297, 0.39927902496563716, 0.6518604856159074, 0.005053044001814127, 0.037824050378368994, -0.629377725123471, 0.0007230654381227738]], "biasesArray": [[0.21359813851886092, -0.11233209533048027, 0.04821286035194456, -0.21915143399673798, 0.12368961683406032, -0.37074217739240256, -0.008788648743365242, -0.12661918977830294, 0.466251607114248, -0.09400484306972744, 0.023790036529178042, 0.08499089062257766, -0.18956132355039082, -0.08257705222392181, 0.07341279379963851, 0.06983063353343692, 0.07951082849039859, 0.2891076239879272, -0.5199218293964237, 0.23033762243049855, -0.10556553578192071, 0.015646183120539732, -0.070292779228458, -0.23053501568639173, -0.44550783557597773, -0.019309146358972594, -0.5254806415713903, -0.3261141014743892, 0.01473108800999562, 0.1932629459313234, -0.3476186331884487, 0.38493742278669224], [-0.5026426521187717, -0.1639436722696205, -0.0935734849288492, 0.004490295570588775, -0.1065551660960308, 0.09874806029171314, 0.05190790627480781, -0.26035856183725037, -0.20468632473311316, -0.009385026192307811, 0.09892707904197785, 0.06338917784584547, -0.018398419064896313, -0.28136956559514675, -0.2459924309087946, -0.18158776317771944, -0.20321883132892976, -0.14468038923957774, -0.19698125166494312, 0.13925511292383955, -0.28482998900002565, -0.09172484757071193, 0.1354369854253279, 0.15682180613783078, -0.12946580182177508, 0.06678535379561631, -0.2658107615127175, 0.1919142051768807, 0.030701326955781087, -0.18834227756890592, 0.4011021953738466, 0.06322503109606974], [-0.08381244096180272, -0.10714637889312573, -0.03919754738127006]], "activationArray": ["elu", "elu", "linear"]}, "meanDictionaryNet": {"weightsArray": [[0.14716183353270787, 0.18147846290138886, -0.23780735290735666, -1.8002887325792356, 0.6597723270887882, 0.9278628434113412, 0.28234627041081367, -0.574918816932533, 0.6835858400773863, -0.07882239488353507, -0.42892019624247474, 0.7848678128824269, -0.5919427604254762, -0.015351880445022852, 0.3114411205886605, -1.5682219732226135, -0.8007725584977736, 0.36514859853019377, -0.3111478007650129, -0.336353331562432, -0.17144134101401248, -0.07668317240167787, 0.09362716970606881, -3.1978516831478467, -0.09043704480028435, 0.13459218584892027, 1.4183180698384854, -0.5747415296800024, -0.4876675865840879, 0.24634152104509213, -0.5611022152477166, 0.9465390172226198], [-0.021397255796528424, -0.010057421699080011, -0.18958891282648715, -0.2232387354412738, -0.25438453057222665, -1.963258268092352, 0.3774629144860234, 0.49672906568192976, -0.844157606818758, 0.6685477224255619, 0.33567619512744373, -0.45894210864976215, -0.27993694579837514, -2.2564877158578582, 0.3147403172517803, 0.30540844768234154, -0.4105258857356779, 2.3603534862610784, -1.0741731982525697, 0.142337788622779, -0.7812586771432617, -0.854287339775322, 0.44797344850381243, -1.4540833748287232, 0.23264754291628967, 0.38541544229587377, 1.3141778210164126, 1.0428569476456793, -0.10981877084046279, 0.7796342405587665, -0.5651642569372424, -0.46631501084760274, 0.12602591201962093, 0.3932320424822254, 0.03242413987874329, -0.247831602359596, 0.08637096249394972, -0.24285400994791945, -0.6998239272644066, -0.34386290029426025, -0.4813937752601364, 0.7570230022760752, 0.3375356085395956, -0.9877220268272924, -0.6807033080388883, -1.5348705334804076, 0.769546661227578, 0.8234583945533125, 1.6718472991798214, 0.12761465267805816, 0.40962562998220464, 0.33404446188775316, -1.1967752729898675, 1.9334736794049567, 0.15250454157609594, -0.5531196252113373, -0.31747987585367893, 0.8738619165463181, 0.39238901576859997, -1.0938257183332478, -0.5238338849259448, -1.4611897740585746, -0.4253758642355851, 0.5977566509691301], [0.6313910726992484, 0.6610427171741486, 0.2524043197341578, 0.3471791870182171, 0.010634932930551753, -0.33655519061553574, 1.4640550965737744, 0.7873745117200226, -0.09833892247592356, 0.9842905940086485, -0.874265167639496, -1.2111253826800694, -0.2966156646242976, -0.49498416636161807, 0.18127462927701996, 0.48004621122299834, -1.991844836801434, 0.006335550884453738, -1.5954652006930885, -0.18637283291842824, -1.8571240140416851, 0.5025455446632312, -0.015028323850604082, -0.13487235190904825]], "biasesArray": [[0.7844708552807017, -0.22133859372055076, -0.2895858663038223, 0.22368561175117632, 0.00317458031827841, 0.5887385030024275, -1.2942958936633377, -0.07061748630314925], [-0.05084670970966344, -0.18292206146514778, -0.263179506916426, -0.028301348614535028, -0.46661409038637897, -0.3135008523288545, 0.9980692086897124, -0.3455163342374444], [0.14841789608516676, 0.008270366800220318, 1.3379068330488584]], "activationArray": ["elu", "elu", "linear"]}, "stdDictionaryNet": {"weightsArray": [[-0.4125617834240716, -0.2331654850012086, 0.6477523729232818, -1.6654455442382232, 2.554243002404381, -1.2692186198879059, -0.7664290041447915, -1.9653596370673077, -0.285889866869249, 0.4104057920938487, 0.9330780956676618, 1.1476288014202594, 3.3892859030427465, -1.8796468919324512, -0.7497200797557603, 1.3070059561508525, -0.35537538792329715, 1.0983293426175205, 1.410156723052895, -0.9951699639239041, 2.1387595298759683, -1.3273442111766804, 1.273117835506016, 1.5188744032763348, -1.0202758301516803, 0.04485908324999588, -0.2544219488270659, 0.7411678830920266, 1.0420027961350575, 0.6685624528143466, 0.24044906533182195, -2.393736011869035], [0.060511439544669574, -1.3858991054688907, -0.5289594968946904, 0.05647076166519143, -0.20540054943348945, 0.2843790432224474, 0.7716988014970985, 0.17590999803447702, -3.0419450119227336, 0.24112932042674082, 1.1606415686596558, 0.38870868040761875, -0.6457095028234803, -0.9853685531385776, -2.226723126214272, -1.445797289920856, 0.050138523627332975, 0.38463583147874963, -0.7565500325827946, -0.8650653631386942, 0.4235649384374719, -0.8116377027317786, 0.2999335645722433, -1.4322847767824405, 0.9284888350257121, 0.21201951272111588, -1.0563073534105112, -0.5075776871346958, 0.5288890016538577, -0.5939849072095779, -0.22411760320400836, -1.629132184585903, 2.3643538343294748, -1.69675844179381, -2.378677097869955, -1.3264452225884082, -2.121303778453894, 0.6001270891117513, -1.0541830005511006, 0.7864332819503614, 1.5297850344456536, -0.3864936040947237, 0.636977321115554, -0.11270703093508909, 0.09100079581426317, -0.3342591913245479, -0.42876050364393425, 0.7620864819868445, -0.8701226213315635, 0.18482477222508556, 0.09211345729101003, 0.11738301227548255, 0.4431574313483271, -0.6751198285684511, 0.1802386681006659, -1.1327524794187258, -0.12921304001038394, 0.10174901361799076, -0.09398655984149362, -0.6288680389670225, 0.7352215962362447, 0.07536787270782785, 0.06228544632466515, -2.290447164252581], [0.3380320539821753, 1.8054870257753113, -0.3710335924483836, -2.0167738774096966, 2.4156257326498767, -0.17285500875280466, 0.4517725572584624, 0.027539933082425954, -1.1878864547952388, -0.09435421931411168, 0.3429244594416727, -0.4241520982115734, -0.5227550569094925, -0.6967921169999087, 0.5824132768453246, -0.27880023639280693, 0.9105344222261597, -0.5856596542814302, -1.1385562502024564, -1.5576632871087137, 1.700664168653571, 1.2519952497136133, 0.07185627455033758, -1.188858684630574]], "biasesArray": [[0.37910360220997547, -0.5304637904231364, 0.6747761364204193, -1.112712542066308, 0.08728307074276584, -1.2748724945897798, 0.06387966387023482, 1.4431918423419756], [0.853049556403387, -0.6478558646073032, -0.3576713100426597, 0.2724447765965392, -0.5697330189439858, 0.2903920912450727, 0.11020721243892143, -0.31431430401887606], [0.17975656125656123, -0.392903790195304, -0.260341452506904]], "activationArray": ["elu", "elu", "softplus"]}};

jcApp.rawdata = {"x=0.010, y=0.000": {"input": [0.01, 0.0], "temperature [degC]": [573, 373, 298, 473, 673, 323, 723, 523, 623, 423], "electrical_resistivity [Ohm m]": [2.85e-05, 6.97e-06, 4.59e-06, 1.4800000000000002e-05, 4.51e-05, 5.08e-06, 5.31e-05, 2.1e-05, 3.68e-05, 1.02e-05], "Seebeck_coefficient [V/K]": [0.000255, 0.00012, 8.340000000000001e-05, 0.000187, 0.000296, 9.54e-05, 0.00031, 0.000223, 0.000281, 0.000151], "thermal_conductivity [W/m/K]": [1.283, 2.918, 3.857, 1.9, 1.032, 3.557, 0.988, 1.533, 1.121, 2.359]}, "x=0.010, y=0.010": {"input": [0.01, 0.01], "temperature [degC]": [723, 523, 298, 573, 673, 373, 323, 473, 423, 623], "electrical_resistivity [Ohm m]": [0.000175, 8.82e-05, 3.43e-05, 0.000112, 0.00015800000000000002, 3.91e-05, 3.44e-05, 6.579999999999999e-05, 4.92e-05, 0.000136], "Seebeck_coefficient [V/K]": [0.000317, 0.000333, 0.000203, 0.000354, 0.000354, 0.000238, 0.000218, 0.000302, 0.000267, 0.000364], "thermal_conductivity [W/m/K]": [1.55116494, 1.431200745, 2.608523775, 1.332160538, 1.380983175, 2.0603153030000003, 2.385334575, 1.57906359, 1.788303465, 1.315421348]}, "x=0.020, y=0.000": {"input": [0.02, 0.0], "temperature [degC]": [373, 723, 673, 573, 623, 423, 323, 298, 523, 473], "electrical_resistivity [Ohm m]": [5.25e-06, 2.62e-05, 2.25e-05, 1.5e-05, 1.87e-05, 6.85e-06, 4.28e-06, 4.03e-06, 1.17e-05, 8.99e-06], "Seebeck_coefficient [V/K]": [9.73e-05, 0.000255, 0.000244, 0.000204, 0.000224, 0.000124, 7.79e-05, 7.09e-05, 0.000179, 0.000153], "thermal_conductivity [W/m/K]": [3.261, 1.247, 1.338, 1.675, 1.475, 2.732, 3.87, 4.199, 1.951, 2.293]}, "x=0.020, y=0.010": {"input": [0.02, 0.01], "temperature [degC]": [298, 623, 323, 673, 523, 573, 373, 723, 473, 423], "electrical_resistivity [Ohm m]": [9.6e-06, 4.95e-05, 8.97e-06, 5.81e-05, 2.78e-05, 3.8e-05, 9.86e-06, 6.15e-05, 1.94e-05, 1.35e-05], "Seebeck_coefficient [V/K]": [9.34e-05, 0.000302, 0.000103, 0.000315, 0.000246, 0.000286, 0.00013700000000000002, 0.000315, 0.000211, 0.000174], "thermal_conductivity [W/m/K]": [2.823318944, 0.987954945, 2.584942088, 0.925949405, 1.321406963, 1.138146143, 2.1550370080000003, 0.917681999, 1.522580495, 1.807805981]}, "x=0.020, y=0.020": {"input": [0.02, 0.02], "temperature [degC]": [323, 473, 298, 623, 423, 723, 673, 373, 573, 523], "electrical_resistivity [Ohm m]": [2.6600000000000006e-05, 3.6e-05, 2.92e-05, 5.8e-05, 3.21e-05, 7.209999999999999e-05, 6.21e-05, 2.62e-05, 4.9e-05, 3.99e-05], "Seebeck_coefficient [V/K]": [0.000122, 0.000233, 0.00011, 0.000321, 0.000196, 0.000334, 0.000327, 0.000154, 0.000299, 0.000265], "thermal_conductivity [W/m/K]": [2.451, 1.481, 2.675, 1.023, 1.723, 0.818, 0.931, 2.0580000000000003, 1.137, 1.3090000000000002]}, "x=0.022, y=0.013": {"input": [0.0225, 0.0125], "temperature [degC]": [723, 573, 523, 323, 423, 373, 473, 298, 673, 623], "electrical_resistivity [Ohm m]": [6.7e-05, 4.25e-05, 3.08e-05, 9.83e-06, 1.6e-05, 1.18e-05, 2.22e-05, 9.77e-06, 6.43e-05, 5.56e-05], "Seebeck_coefficient [V/K]": [0.000308, 0.000263, 0.000231, 9.98e-05, 0.000164, 0.000128, 0.0002, 9.08e-05, 0.000304, 0.00029], "thermal_conductivity [W/m/K]": [1.008732711, 1.250992805, 1.479565889, 2.926282952, 2.05305165, 2.458187176, 1.769730522, 3.204129275, 1.026525825, 1.093592179]}, "x=0.022, y=0.018": {"input": [0.0225, 0.0175], "temperature [degC]": [573, 673, 623, 723, 523, 323, 373, 473, 423, 298], "electrical_resistivity [Ohm m]": [3.8e-05, 5.6299999999999986e-05, 4.89e-05, 6.19e-05, 2.68e-05, 1.24e-05, 1.17e-05, 1.99e-05, 1.44e-05, 1.41e-05], "Seebeck_coefficient [V/K]": [0.000279, 0.00031600000000000004, 0.000303, 0.00032, 0.000243, 0.000112, 0.000139, 0.000211, 0.000176, 0.000103], "thermal_conductivity [W/m/K]": [1.195760797, 0.932882155, 1.033989325, 0.8937873829999999, 1.392582755, 2.724501208, 2.267496799, 1.63928425, 1.8994667, 2.9563736510000003]}, "x=0.028, y=0.013": {"input": [0.0275, 0.0125], "temperature [degC]": [673, 723, 373, 473, 298, 323, 573, 623, 523, 423], "electrical_resistivity [Ohm m]": [4.370000000000001e-05, 4.58e-05, 7.96e-06, 1.46e-05, 6.42e-06, 6.55e-06, 2.7300000000000006e-05, 4.04e-05, 1.96e-05, 1.07e-05], "Seebeck_coefficient [V/K]": [0.000292, 0.000296, 0.000124, 0.00019, 8.77e-05, 9.59e-05, 0.000255, 0.000294, 0.000219, 0.00015800000000000002], "thermal_conductivity [W/m/K]": [0.8811150520000001, 0.7537087659999999, 2.4555885230000003, 1.717973183, 3.190521625, 2.930344578, 1.142633218, 0.953535467, 1.472548442, 2.050570645]}, "x=0.028, y=0.018": {"input": [0.0275, 0.0175], "temperature [degC]": [573, 423, 523, 723, 323, 298, 673, 623, 373, 473], "electrical_resistivity [Ohm m]": [3.2e-05, 1.15e-05, 2.21e-05, 4.86e-05, 6.76e-06, 6.28e-06, 5.04e-05, 5.1e-05, 8.64e-06, 1.55e-05], "Seebeck_coefficient [V/K]": [0.000261, 0.000157, 0.000224, 0.0003, 0.0001, 9.32e-05, 0.000299, 0.00031, 0.000127, 0.000187], "thermal_conductivity [W/m/K]": [1.281744522, 2.306874218, 1.599521432, 0.873554099, 3.254886504, 3.4822499, 0.979922939, 1.058369958, 2.733679188, 1.958516267]}, "x=0.030, y=0.000": {"input": [0.03, 0.0], "temperature [degC]": [298, 623, 573, 523, 673, 473, 323, 723, 423, 373], "electrical_resistivity [Ohm m]": [3.65e-06, 1.5399999999999998e-05, 1.26e-05, 1.01e-05, 1.82e-05, 7.95e-06, 3.94e-06, 2.08e-05, 6.19e-06, 4.85e-06], "Seebeck_coefficient [V/K]": [7.7e-05, 0.000212, 0.000193, 0.000171, 0.000228, 0.000149, 8.96e-05, 0.000238, 0.000125, 0.000104], "thermal_conductivity [W/m/K]": [4.533, 1.661, 1.885, 2.184, 1.49, 2.563, 4.25, 1.378, 3.04, 3.611]}, "x=0.030, y=0.005": {"input": [0.03, 0.005], "temperature [degC]": [723, 623, 298, 423, 523, 573, 373, 473, 323, 673], "electrical_resistivity [Ohm m]": [3.66e-05, 2.69e-05, 5.81e-06, 8.41e-06, 1.59e-05, 2.11e-05, 6.34e-06, 1.16e-05, 5.56e-06, 3.2500000000000004e-05], "Seebeck_coefficient [V/K]": [0.000291, 0.000263, 8.14e-05, 0.000147, 0.000207, 0.000241, 0.000117, 0.00018, 9.16e-05, 0.000281], "thermal_conductivity [W/m/K]": [1.052, 1.23, 3.765, 2.391, 1.685, 1.436, 2.8960000000000004, 1.99, 3.468, 1.102]}, "x=0.030, y=0.020": {"input": [0.03, 0.02], "temperature [degC]": [673, 573, 323, 373, 623, 523, 298, 423, 723, 473], "electrical_resistivity [Ohm m]": [4.59e-05, 3.41e-05, 7.39e-06, 9.36e-06, 4.2700000000000015e-05, 2.46e-05, 6.97e-06, 1.28e-05, 4.820000000000001e-05, 1.76e-05], "Seebeck_coefficient [V/K]": [0.000305, 0.000278, 0.000106, 0.000136, 0.000298, 0.000243, 9.62e-05, 0.000171, 0.000312, 0.000207], "thermal_conductivity [W/m/K]": [0.905, 1.166, 2.676, 2.246, 1.018, 1.36, 2.886, 1.882, 0.79, 1.592]}, "x=0.030, y=0.030": {"input": [0.03, 0.03], "temperature [degC]": [573, 473, 523, 723, 298, 623, 373, 673, 423, 323], "electrical_resistivity [Ohm m]": [5.59e-05, 3.16e-05, 4.11e-05, 7.340000000000001e-05, 2.54e-05, 6.36e-05, 2.17e-05, 6.579999999999999e-05, 2.49e-05, 2.29e-05], "Seebeck_coefficient [V/K]": [0.000303, 0.000234, 0.00027, 0.000332, 0.000112, 0.000313, 0.000156, 0.000319, 0.000192, 0.000122], "thermal_conductivity [W/m/K]": [1.14866249, 1.491197771, 1.309612802, 0.8528990940000001, 2.694886008, 1.034484062, 2.0744707, 0.928559498, 1.747067499, 2.474783016]}, "x=0.032, y=0.014": {"input": [0.032, 0.014], "temperature [degC]": [723, 373, 423, 323, 573, 673, 523, 473, 298, 623], "electrical_resistivity [Ohm m]": [4.23e-05, 7.82e-06, 1.03e-05, 6.53e-06, 2.54e-05, 4.28e-05, 1.82e-05, 1.39e-05, 6.4e-06, 4.120000000000001e-05], "Seebeck_coefficient [V/K]": [0.000292, 0.000124, 0.000154, 9.8e-05, 0.00024700000000000004, 0.00028900000000000003, 0.000213, 0.000186, 9.09e-05, 0.000299], "thermal_conductivity [W/m/K]": [0.916187271, 2.574636876, 2.151465177, 3.082716813, 1.254450733, 0.957271902, 1.569432904, 1.82826608, 3.348397427, 1.044919115]}, "x=0.033, y=0.013": {"input": [0.0325, 0.0125], "temperature [degC]": [623, 673, 298, 473, 573, 423, 373, 323, 723, 523], "electrical_resistivity [Ohm m]": [3.33e-05, 3.69e-05, 5.44e-06, 1.19e-05, 2.19e-05, 9.1e-06, 6.99e-06, 5.72e-06, 3.7e-05, 1.5399999999999998e-05], "Seebeck_coefficient [V/K]": [0.000272, 0.000276, 8.62e-05, 0.00017, 0.000233, 0.000143, 0.000115, 9.15e-05, 0.00028, 0.000196], "thermal_conductivity [W/m/K]": [1.290319967, 1.163085844, 3.8709599, 2.203086503, 1.530958417, 2.587554832, 3.064682794, 3.590215042, 1.095319843, 1.890533113]}, "x=0.033, y=0.018": {"input": [0.0325, 0.0175], "temperature [degC]": [473, 623, 373, 723, 298, 573, 423, 323, 673, 523], "electrical_resistivity [Ohm m]": [1.4300000000000004e-05, 4.54e-05, 7.96e-06, 4.39e-05, 5.76e-06, 2.92e-05, 1.06e-05, 6.21e-06, 4.54e-05, 2.03e-05], "Seebeck_coefficient [V/K]": [0.000187, 0.000302, 0.000126, 0.000295, 9.24e-05, 0.000259, 0.000157, 9.95e-05, 0.000296, 0.000223], "thermal_conductivity [W/m/K]": [1.73002068, 0.93939021, 2.458667925, 0.83470743, 3.19006998, 1.1294721, 2.06886231, 2.907701955, 0.877406985, 1.417349745]}, "x=0.037, y=0.013": {"input": [0.0375, 0.0125], "temperature [degC]": [623, 523, 373, 323, 573, 723, 298, 423, 473, 673], "electrical_resistivity [Ohm m]": [2.64e-05, 1.34e-05, 6.47e-06, 5.36e-06, 1.85e-05, 3.16e-05, 5.11e-06, 8.26e-06, 1.06e-05, 3.02e-05], "Seebeck_coefficient [V/K]": [0.000251, 0.000187, 0.000109, 8.83e-05, 0.00022, 0.000266, 8.290000000000001e-05, 0.000135, 0.000161, 0.000262], "thermal_conductivity [W/m/K]": [1.265174485, 1.768774307, 2.847328695, 3.335834245, 1.451794583, 1.034643776, 3.578714813, 2.419200236, 2.063798726, 1.129326032]}, "x=0.037, y=0.018": {"input": [0.0375, 0.0175], "temperature [degC]": [373, 523, 623, 573, 298, 673, 323, 473, 723, 423], "electrical_resistivity [Ohm m]": [7.54e-06, 1.7899999999999998e-05, 4.13e-05, 2.6e-05, 5.52e-06, 4.19e-05, 5.94e-06, 1.3e-05, 3.99e-05, 9.88e-06], "Seebeck_coefficient [V/K]": [0.00012, 0.000208, 0.000291, 0.000245, 8.78e-05, 0.000284, 9.6e-05, 0.000175, 0.00028, 0.000146], "thermal_conductivity [W/m/K]": [2.717389472, 1.628249945, 1.160111025, 1.34163428, 3.442117653, 1.075491162, 3.203271266, 1.953081032, 1.01407352, 2.312033031]}, "x=0.040, y=0.000": {"input": [0.04, 0.0], "temperature [degC]": [373, 423, 523, 573, 298, 723, 623, 673, 473, 323], "electrical_resistivity [Ohm m]": [5.05e-06, 6.46e-06, 1.06e-05, 1.33e-05, 3.85e-06, 2.1600000000000007e-05, 1.62e-05, 1.9e-05, 8.32e-06, 4.11e-06], "Seebeck_coefficient [V/K]": [0.0001, 0.000123, 0.000171, 0.000193, 7.54e-05, 0.000237, 0.000212, 0.000227, 0.000149, 8.28e-05], "thermal_conductivity [W/m/K]": [3.42, 2.872, 2.067, 1.776, 4.33, 1.211, 1.534, 1.345, 2.425, 4.0360000000000005]}, "x=0.040, y=0.005": {"input": [0.04, 0.005], "temperature [degC]": [423, 473, 373, 723, 298, 323, 623, 673, 573, 523], "electrical_resistivity [Ohm m]": [8.68e-06, 1.2e-05, 6.57e-06, 3.67e-05, 5.95e-06, 5.75e-06, 2.7399999999999995e-05, 3.2899999999999987e-05, 2.15e-05, 1.63e-05], "Seebeck_coefficient [V/K]": [0.000146, 0.00018, 0.000113, 0.00028700000000000004, 7.92e-05, 8.93e-05, 0.000262, 0.00028, 0.00024, 0.00021], "thermal_conductivity [W/m/K]": [2.634, 2.1630000000000003, 3.215, 1.149, 4.269, 3.872, 1.356, 1.208, 1.569, 1.825]}, "x=0.040, y=0.010": {"input": [0.04, 0.01], "temperature [degC]": [373, 673, 423, 473, 723, 298, 623, 323, 523, 573], "electrical_resistivity [Ohm m]": [5.23e-06, 2.63e-05, 6.77e-06, 8.77e-06, 2.8600000000000007e-05, 4e-06, 2.3e-05, 4.28e-06, 1.14e-05, 1.6e-05], "Seebeck_coefficient [V/K]": [0.000111, 0.000261, 0.000134, 0.00015900000000000002, 0.000268, 8.4e-05, 0.00024900000000000004, 9.53e-05, 0.000184, 0.000217], "thermal_conductivity [W/m/K]": [2.792, 1.074, 2.334, 1.996, 0.848, 3.5860000000000003, 1.166, 3.324, 1.6669999999999998, 1.368]}, "x=0.040, y=0.020": {"input": [0.04, 0.02], "temperature [degC]": [673, 723, 323, 423, 523, 573, 373, 623, 473, 298], "electrical_resistivity [Ohm m]": [3.3600000000000004e-05, 3.3100000000000005e-05, 4.42e-06, 7.42e-06, 1.39e-05, 2.07e-05, 5.59e-06, 3.3e-05, 9.97e-06, 4.09e-06], "Seebeck_coefficient [V/K]": [0.000279, 0.000278, 9.48e-05, 0.000143, 0.000201, 0.000236, 0.000115, 0.000276, 0.00017, 8.85e-05], "thermal_conductivity [W/m/K]": [1.031, 0.934, 3.168, 2.305, 1.609, 1.29, 2.719, 1.135, 1.935, 3.41]}, "x=0.040, y=0.030": {"input": [0.04, 0.03], "temperature [degC]": [423, 373, 673, 523, 723, 298, 323, 623, 573, 473], "electrical_resistivity [Ohm m]": [1.4800000000000002e-05, 1.05e-05, 4.99e-05, 2.93e-05, 4.68e-05, 7.000000000000001e-06, 7.76e-06, 4.88e-05, 4.01e-05, 2.08e-05], "Seebeck_coefficient [V/K]": [0.000173, 0.00013700000000000002, 0.000307, 0.000246, 0.000302, 9.83e-05, 0.000105, 0.000301, 0.000282, 0.000209], "thermal_conductivity [W/m/K]": [1.806, 2.15, 0.867, 1.306, 0.69, 2.769, 2.576, 0.979, 1.116, 1.525]}, "x=0.040, y=0.040": {"input": [0.04, 0.04], "temperature [degC]": [473, 673, 623, 423, 373, 323, 573, 523, 298, 723], "electrical_resistivity [Ohm m]": [3.98e-05, 8.05e-05, 7e-05, 3.83e-05, 3.5600000000000005e-05, 3.2200000000000003e-05, 6.13e-05, 4.4e-05, 3.2500000000000004e-05, 0.000118], "Seebeck_coefficient [V/K]": [0.000243, 0.000343, 0.000328, 0.000224, 0.000187, 0.000152, 0.000315, 0.000273, 0.000139, 0.000352], "thermal_conductivity [W/m/K]": [1.25, 0.838, 0.926, 1.399, 1.631, 1.963, 1.009, 1.126, 2.143, 0.759]}, "x=0.040, y=0.050": {"input": [0.04, 0.05], "temperature [degC]": [623, 298, 573, 523, 723, 673, 323, 423, 373, 473], "electrical_resistivity [Ohm m]": [9.33e-05, 3.4899999999999995e-05, 8.03e-05, 6.08e-05, 0.000151, 0.000108, 3.13e-05, 4.49e-05, 3.43e-05, 4.7e-05], "Seebeck_coefficient [V/K]": [0.000346, 0.00012, 0.000332, 0.000302, 0.000342, 0.000349, 0.000132, 0.00023, 0.000178, 0.000261], "thermal_conductivity [W/m/K]": [0.908, 2.021, 0.997, 1.117, 0.755, 0.812, 1.86, 1.3730000000000002, 1.5730000000000002, 1.254]}, "x=0.040, y=0.100": {"input": [0.04, 0.1], "temperature [degC]": [298, 673, 573, 473, 523, 423, 373, 623, 323, 723], "electrical_resistivity [Ohm m]": [0.00113, 0.000115, 0.00028, 0.0007469999999999999, 0.000435, 0.00136, 0.00225, 0.000173, 0.00168, 9.33e-05], "Seebeck_coefficient [V/K]": [0.000173, -0.000254, -0.000234, -0.000185, -0.000221, -0.000126, -7.1e-06, -0.00026000000000000003, 0.000145, -0.000253], "thermal_conductivity [W/m/K]": [1.6869999999999998, 0.907, 1.024, 1.14, 1.08, 1.238, 1.378, 0.967, 1.564, 0.8270000000000001]}, "x=0.040, y=0.150": {"input": [0.04, 0.15], "temperature [degC]": [373, 298, 323, 623, 523, 573, 673, 723, 423, 473], "electrical_resistivity [Ohm m]": [0.00101, 0.000698, 0.0008380000000000001, 0.00012, 0.000278, 0.00019, 8.27e-05, 7.03e-05, 0.000701, 0.000436], "Seebeck_coefficient [V/K]": [8.96e-06, 8.65e-05, 7.35e-05, -0.000225, -0.000165, -0.000189, -0.000232, -0.000233, -7.31e-05, -0.00013000000000000002], "thermal_conductivity [W/m/K]": [1.5219999999999998, 1.779, 1.67, 1.097, 1.218, 1.148, 1.033, 0.973, 1.395, 1.283]}, "x=0.050, y=0.000": {"input": [0.05, 0.0], "temperature [degC]": [673, 473, 723, 623, 373, 323, 423, 573, 298, 523], "electrical_resistivity [Ohm m]": [1.93e-05, 8.16e-06, 2.2e-05, 1.65e-05, 4.86e-06, 3.93e-06, 6.27e-06, 1.34e-05, 3.62e-06, 1.06e-05], "Seebeck_coefficient [V/K]": [0.000236, 0.000151, 0.000246, 0.00022, 0.0001, 8.070000000000001e-05, 0.000125, 0.000199, 7.159999999999999e-05, 0.000176], "thermal_conductivity [W/m/K]": [1.395, 2.405, 1.264, 1.56, 3.409, 4.049, 2.86, 1.775, 4.357, 2.047]}, "x=0.050, y=0.005": {"input": [0.05, 0.005], "temperature [degC]": [673, 573, 298, 423, 323, 473, 373, 523, 723, 623], "electrical_resistivity [Ohm m]": [3.69e-05, 2.4e-05, 9.33e-06, 9.67e-06, 7.92e-06, 1.33e-05, 7.68e-06, 1.8100000000000006e-05, 3.99e-05, 3.09e-05], "Seebeck_coefficient [V/K]": [0.000292, 0.00024900000000000004, 7.55e-05, 0.000149, 8.41e-05, 0.000185, 0.000112, 0.00022, 0.000299, 0.00027400000000000005], "thermal_conductivity [W/m/K]": [1.053, 1.346, 3.651, 2.296, 3.358, 1.896, 5.566, 1.6, 0.978, 1.163]}, "x=0.050, y=0.020": {"input": [0.05, 0.02], "temperature [degC]": [723, 523, 473, 323, 373, 623, 423, 673, 573, 298], "electrical_resistivity [Ohm m]": [4.41e-05, 2e-05, 1.49e-05, 6.08e-06, 7.82e-06, 3.84e-05, 1.0800000000000003e-05, 4.29e-05, 2.7600000000000007e-05, 5.63e-06], "Seebeck_coefficient [V/K]": [0.000302, 0.000227, 0.000195, 0.000101, 0.000128, 0.000291, 0.000162, 0.000297, 0.000258, 9.19e-05], "thermal_conductivity [W/m/K]": [0.7809999999999999, 1.367, 1.634, 2.784, 2.323, 0.921, 1.928, 0.831, 1.099, 3.027]}, "x=0.060, y=0.000": {"input": [0.06, 0.0], "temperature [degC]": [523, 423, 573, 298, 323, 473, 673, 723, 373, 623], "electrical_resistivity [Ohm m]": [1.51e-05, 8.27e-06, 1.95e-05, 4.4e-06, 4.76e-06, 1.13e-05, 2.39e-05, 2.42e-05, 6.09e-06, 2.31e-05], "Seebeck_coefficient [V/K]": [0.000201, 0.000138, 0.000225, 7.35e-05, 8.159999999999999e-05, 0.000168, 0.000245, 0.00024700000000000004, 0.000106, 0.000241], "thermal_conductivity [W/m/K]": [1.803, 2.543, 1.561, 3.978, 3.659, 2.117, 1.3, 1.217, 3.063, 1.412]}};