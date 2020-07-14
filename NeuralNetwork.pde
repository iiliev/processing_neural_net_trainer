import java.util.Map;

class NeuralNetwork{  
  Map<String,Matrix> parameters;
  String[] activations;
  int L;            //number of layers
  float learningRate = 0.0075;//0.01; //0.0075;
  int inputSize;
  int outputSize;
  
  NeuralNetwork( int[] dims ){
    parameters = initializeParameters( dims );
    L = dims.length;
    inputSize  = dims[0];
    outputSize = dims[dims.length-1];
    activations = new String[L-1];
    for( int i = 0; i < L-1; i++ ) activations[i] = "sigmoid"; //set default
    //printMap(parameters);
  }
  
  NeuralNetwork( NeuralNetwork other ){
    parameters = new HashMap<String,Matrix>();
    activations = new String[other.activations.length];
    for (Map.Entry me : other.parameters.entrySet())
      parameters.put( (String)me.getKey(), new Matrix((Matrix)(me.getValue())));
    for( int i = 0; i < other.activations.length; i++ )
      activations[i] = other.activations[i];
    L = activations.length + 1;
    inputSize = parameters.get( "W1" ).cols;
    outputSize = parameters.get( "B" + Integer.toString( L - 1 ) ).rows;    
  }
  
  NeuralNetwork( String fileName ){
    MessageFormat loadSavePathFormatter = new MessageFormat("data/{0}.json");
    JSONObject json = loadJSONObject(loadSavePathFormatter.format(new Object[] {fileName}));    
    fromJSON( json );
  }
  
  NeuralNetwork( JSONObject json ){
    fromJSON( json );    
  }

  Matrix predict(float[] input){
    Matrix X = transpose(new Matrix( new float[][]{input} ));
    Matrix resp = transpose(predict(X));
    return resp; //it is a row matrix now; use .data[0] to get an array; otherwise keep a matrix so you can do .bin()
  }
  
  Matrix predict(Matrix input){
    return lModelForward( input, parameters, activations )[L-2].get("A");
  }
  
  float train(float[] input, float[] target){
    Matrix X = transpose(new Matrix( new float[][]{input } )); //we need two column vectors
    Matrix Y = transpose(new Matrix( new float[][]{target} ));
    return train( X, Y );
  }
    
  float train(Matrix input, Matrix target){ //the columns are the input/target examples
    Map<String,Matrix>[] caches = lModelForward( input, parameters, activations ); 
    Matrix AL = caches[L-2].get("A");
    Map<String,Matrix> grads = lModelBackward( AL, target, caches, activations );

    //if( isNaN( grads ) || isInfinite( grads ) ){ printMap( grads );printMap( parameters );throw new RuntimeException();}    
    
    updateParameters( parameters, grads, learningRate );

    //if( Float.isNaN(cost) || Float.isInfinite(cost)){println(cost);printArr(target);X.printM();Y.printM();
    //  AL.printM();printMap( grads );printMap( parameters );throw new RuntimeException();      }
    
    return RMSE( AL, target );
  }

  void fromJSON( JSONObject json ){
    JSONObject weights = json.getJSONObject("weights");
    parameters = new HashMap<String,Matrix>(); 
    String[] keys = (String[]) weights.keys().toArray(new String[weights.size()]);
    for( int i = 0; i < keys.length; i++ ){
      parameters.put(keys[i],new Matrix(weights.getJSONArray(keys[i]))); 
    }
    JSONArray  jsonActivations = json.getJSONArray("activations");
    activations = new String[jsonActivations.size()];
    for( int i = 0; i < activations.length; i++ )
      activations[i] = jsonActivations.getString(i);
    L = activations.length + 1;
    inputSize = parameters.get( "W1" ).cols;
    outputSize = parameters.get( "B" + Integer.toString( L - 1 ) ).rows;    
  }

  JSONObject toJSON(){
    JSONObject weights = new JSONObject();
    for (Map.Entry me : parameters.entrySet())
      weights.setJSONArray( (String)me.getKey(), ((Matrix)(me.getValue())).toJSON());
    JSONArray jsonActivations = new JSONArray();
    for( int i = 0; i < activations.length; i++ )
      jsonActivations.setString(i,activations[i]);
    JSONObject json = new JSONObject();
    json.setJSONObject("weights",weights);
    json.setJSONArray("activations",jsonActivations);
    return json;    
  }

  void save(String fileName) {
    MessageFormat loadSavePathFormatter = new MessageFormat("data/{0}.json");
    saveJSONObject(toJSON(), loadSavePathFormatter.format(new Object[] {fileName}));
  }  
  
}

Map<String,Matrix> initializeParameters( int[] dims ){
  Map<String,Matrix> params = new HashMap<String,Matrix> ();
  int L = dims.length;
  for( int l = 1; l < L; l++ ){
    params.put( "W"+Integer.toString(l), new Matrix(dims[l], dims[l-1]).randomize().multiply(sqrt(2./dims[l-1]))); //Ng 
    params.put( "B"+Integer.toString(l), new Matrix(dims[l],         1).randomize().multiply(.01));      
  }
  return params;
}

Map<String,Matrix> linearForward(Matrix A_prev, Matrix W, Matrix B){
  Map<String,Matrix> response = new HashMap<String,Matrix>();
  Matrix Z = multiply( W, A_prev ).broadcastAndAdd(B);
  response.put("Z",Z);  
  response.put("A_prev",A_prev);  
  response.put("W",W);  
  response.put("B",B);    
    //if( Z.isInfinite()||Z.isNaN() ){println("linearForward A_prev");A_prev.printM();println("W");W.printM();println("B");
    //  B.printM();println("Z");Z.printM();println();throw new RuntimeException();}    
  return response;
}

Map<String,Matrix> linearActivationForward( Matrix A_prev, Matrix W, Matrix B, String activation){
  Map<String,Matrix> response = linearForward( A_prev, W, B );
  Matrix Z = response.get("Z");
  Matrix A=null;
  switch( activation ){
    case "sigmoid" : A = sigmoid(Z); break;  
    case "relu"    : A =    relu(Z); break;  
    case "tanh"    : A =    tanh(Z); break;  
    default        : A = sigmoid(Z);  
  }
  response.put("A",A);
  return response;
}

Map<String,Matrix>[] lModelForward(Matrix X, Map<String,Matrix> parameters, String[] activations ){
  int L = (int)(parameters.size()/2);
  Map<String,Matrix>[] caches = (Map<String,Matrix>[]) new Map[L]; //weird cast
  Matrix A_prev = X;
  for( int i = 0; i < L; i++ ){
    Map<String,Matrix> cache = linearActivationForward( A_prev, 
                                                        parameters.get("W"+Integer.toString(i+1)),
                                                        parameters.get("B"+Integer.toString(i+1)),
                                                        activations[i] );
    //if( isInfinite( cache ) || isNaN( cache )){println(i);printMap( cache );println();
    //  printMap( parameters );throw new RuntimeException();}       
    caches[i] = cache;                                                            
    A_prev = cache.get("A");                                    
  }
  return caches;
}

Map<String,Matrix> linearBackward(Matrix dZ, Matrix A_prev, Matrix W){
  Map<String,Matrix> response = new HashMap<String,Matrix>();
  int m = A_prev.cols;
  Matrix dW = multiply( dZ, transpose(A_prev) ).multiply(1./m);
  //dW.printM();
  Matrix dB = rowWiseSum( dZ ).multiply(1./m);
  Matrix dA_prev = multiply( transpose(W), dZ );
  
  //if( dA_prev.isInfinite() || dA_prev.isNaN() ){println("linearBackward W");W.printM();println("dZ");
  //  dZ.printM();println("dA_prev");dA_prev.printM();throw new RuntimeException();}
  
  response.put("dA_prev", dA_prev);
  response.put("dW", dW);
  response.put("dB", dB);
  return response;
}

Map<String,Matrix> linearActivationBackward( Matrix dA, Matrix Z, Matrix A_prev, Matrix W, String activation){
  Matrix dZ = null;
  switch( activation ){
    case "relu"   : dZ =    reluBackward( dA, Z ); break;
    case "sigmoid": dZ = sigmoidBackward( dA, Z ); break;
    case "tanh":    dZ =    tanhBackward( dA, Z ); break;
    default: dZ = sigmoidBackward( dA, Z );
  }
  return linearBackward( dZ, A_prev, W );
}

Map<String,Matrix> lModelBackward( Matrix AL, Matrix Y, Map<String,Matrix>[] caches, String[] activations ){ //Y - actual
  Map<String,Matrix> grads = new HashMap<String,Matrix>();
  int L = caches.length; //the number of layers
  Matrix dAL = elementWiseDivide(add( Y , -1 ).multiply(-1), //this may blow up when things are close to 0, see elementWiseDivide
                                 add( AL, -1 ).multiply(-1)).subtract(elementWiseDivide( Y, AL )); // (1-Y)/(1-AL) - Y/AL 

    //if( dAL.isInfinite() || dAL.isNaN() ){println("lModelBackward dAL");dAL.printM();println("Y");Y.printM();
    //  println("AL");AL.printM();println();throw new RuntimeException();}
    
  Matrix current_dA = dAL;
  for( int l = L-1; l >= 0; l-- ){
    Map<String,Matrix> currentCache = caches[l];
    
    //currentCache.get("A_prev").printM();
    
    Map<String,Matrix> lab = linearActivationBackward(  current_dA, 
                                                        currentCache.get("Z"), 
                                                        currentCache.get("A_prev"), 
                                                        currentCache.get("W"),
                                                        activations[l] );
    grads.put("dA" + Integer.toString(l  ), lab.get("dA_prev"));                                                 
    grads.put("dW" + Integer.toString(l+1), lab.get("dW"     ));                                                 
    grads.put("dB" + Integer.toString(l+1), lab.get("dB"     ));   
    
    current_dA = lab.get("dA_prev");
    //if( isNaN( grads ) || isInfinite( grads ) ){ printMap( grads );throw new RuntimeException();}
  }
  return grads;
}

Map<String,Matrix> updateParameters(Map<String,Matrix> parameters, Map<String,Matrix> grads, float learningRate ){
  int L = parameters.size() / 2;
  for( int l = 0; l < L; l++ ){
    String index = Integer.toString(l+1);  
    parameters.get("W"+index).subtract((grads.get("dW"+index).multiply( learningRate )));  
    parameters.get("B"+index).subtract((grads.get("dB"+index).multiply( learningRate )));  
  }
  return parameters;
}

void printMap( Map<String,Matrix> m ){
  for (Map.Entry me : m.entrySet()) {
    println(me.getKey());
    ((Matrix)(me.getValue())).printM();
  }  
}

boolean isNaN( Map<String,Matrix> map ){
  for (Map.Entry me : map.entrySet())
    if(((Matrix)(me.getValue())).isNaN()){
      println( me.getKey() + " is NaN" );
      return true;
    }
  return false;
}

boolean isInfinite( Map<String,Matrix> map ){
  for (Map.Entry me : map.entrySet())
    if(((Matrix)(me.getValue())).isInfinite()){
      println( me.getKey() + " is infinite" );
      return true;
    }
  return false;
}
