const tf = require('@tensorflow/tfjs');
const fs = require('fs');

try {
    let arquivo = fs.readFileSync('cotacao-do-dolar.csv', {encoding: 'utf8'});
    arquivo = arquivo.toString().trim();
    
    const linhas = arquivo.split('\r\n');
    const X = [];
    const Y = [];
    
    let qtdLinhas = 0;
    
    for (let i = 1; i < linhas.length; i++) {
      
        let celulas1 = [];
        if(qtdLinhas == (linhas.length -2) ){
            celulas1 = ['31.12.2019', 3.8813, 3.8813, 3.8813, 3.8813];
        }else{
            celulas1 = linhas[i + 1].split(';');
        }
        const celulas2 = linhas[i].split(';');
    
        const fechamentoX = Number(celulas1[1]);
        const aberturaX = Number(celulas1[2]);
        const maximaX = Number(celulas1[3]);
        const minimaX = Number(celulas1[4]);
    
        const fechamentoY = Number(celulas2[1]);
        const aberturaY = Number(celulas2[2]);
        const maximaY = Number(celulas2[3]);
        const minimaY = Number(celulas2[4]);
    
        X.push([fechamentoX, aberturaX, maximaX, minimaX]);
    
        Y.push([fechamentoY, aberturaY, maximaY, minimaY]);
        qtdLinhas++;
    }
    
    const model = tf.sequential();
    const inputLayer = tf.layers.dense({units: 4, inputShape: [4]});
    model.add(inputLayer);
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    
    const x = tf.tensor(X, [qtdLinhas, 4]);
    const y = tf.tensor(Y);
    
    const arrInput = [ [3.9285, 3.9708, 3.9781, 3.9251] ]; //08.05.2019
    //const arrInput = [ [3.9466, 3.9283, 3.9818, 3.9283] ]
    const input = tf.tensor(arrInput, [1, 4]);
    
    model.fit(x, y, {epochs: 1000}).then((history) => {
        console.log(`history ->${history}`);
        let output = model.predict(input).dataSync();
        console.log('Preço das cotções:');
        console.log(`Fechamento R$ ->${Number(output[0].toFixed(4))}`);
        console.log(`Abertura R$ ->${Number(output[1].toFixed(4))}`);
        console.log(`Máxima R$ ->${Number(output[2].toFixed(4))}`);
        console.log(`Minima R$ ->${Number(output[3].toFixed(4))}`);
    }); 
} catch (error) {
    console.log(`error ->${error}`);
}

