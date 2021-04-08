async function learnLinear(){
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));//Criação da Rede Neural.

  model.compile({//tipo de perda e otimizador.
   loss: 'meanSquaredError',
   optimizer: 'sgd'
  });

  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);//treinamento do modelo.
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);
  
  await model.fit(xs, ys, {epochs: 500});//treinamento.
  
  document.getElementById('output_field').innerText =
   model.predict(tf.tensor2d([10], [1, 1]));//descobre o Y para X =10.
 }

 //o algoritimo não sabe a formula, ele aprendeu de acordo com os dados fornecidos.
 learnLinear();
 