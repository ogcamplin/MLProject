#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include <chrono>


using namespace std;

// activation function
double activation(double x)
{
    return 1.0/(1.0 + exp(-x));
}

// derivative of activation function
double d_activation(double x){
    return (1.0 - activation(x))*activation(x);
 }  
 

// 
void Randomize(double* a, int n, double min, double max){
	//cout<<"RAND_MAX="<<RAND_MAX<<endl;
    srand (time(NULL));
    for ( int i = 0 ; i < n ; i++){
        double f = (double)rand() / RAND_MAX;
        a[i]= min + f * (max - min);
    }
}

void PrintVector(double* a, int n){
    for ( int i = 0 ; i < n ; i++){
        cout<<a[i]<<" ";
    }
    cout<<endl;
}


class NN{
//	friend class MatrixD;
//	friend class VectorD;
	
	private:
	  int nInputs; // number of inputs
	  int nOutputs; //number of outputs
	  int nHiddenNeurons; // number of neurons in hidden layer
	 
	  // search parameters
	  double dw;   // step for gradient estimation
	  double learningRate; // hm, learning  rate
	  int nStepsMax;
	  
	  // whole training set 
	  double* inputsTraining;   // inputs - column, pattern - row 
	  double* outputsTraining;  // outputs - column, pattern - row
	  int nTrainingEntries;  // number of rows in training set
	  
	  // current NN situation
	  double* currentInputs;  // row picked from inputsTraining
	  double* currentAnswers; // ditto for training answers 
	  double* currentOutputs; // guess what? 
	  double* currentError;   // current differnce between answers and output
	  //output error for current dataset row
	  double sumOfOutputErrors;
	  // sum of errors for all dataset entries
	  double totalDatasetError;
	  
	  
	  // input to hidden layer
	  double* weightsHidden;  // hidden layer weights matrix Wh
	  double* biasHidden;     // hidden layer bias vector bh
	  double* d_weightsHidden;  // hidden layer weights matrix Wh derivative
	  double* d_biasHidden;     // hidden layer bias vector bh derivative
	  double* deltaHidden;  // for backpropagation
	  
	  // state of the hidden layer
	  double* netHidden;     // yh = Wh*z + bh - renaming 
	  double* outHidden;   //  zh = activ(Wh*x + bh) - layer output
	  
      // hidden layer to output layer
      double* weightsOutput;  // Wh
      double* biasOutput;     // bh
      double* d_weightsOutput;  // dWh/dw - gradients
      double* d_biasOutput;     // dbh / dw - gradients
      double* deltaOutput;     // dbh / dw - gradients
 
	  // state of the output layer
      double* nnSumOutput;  
      double* netOutput;  // yo = Wo*zh + bo
      double* nnBiaOutput;
	  double* nnOutOutput; // zo = activ(yo)
	  
	public:
	  NN(){ }; // constructor
	  int LoadTrainingSet(string file,int nInp,int nHid , int nOut);
	  int LoadWorkingSet(string file, int nInp,int nHid , int nOut);
	  void DisplayDigit(int nImage); // ASCII art display
	  int InitNet(double min, double max);
	  void GetTrainingEntry(int iTrainRow);
	  void ForwardProp();
	  double GetTotalError(){ return sumOfOutputErrors;};
	  void PrintErr(){ cout<<" Error: "<<sumOfOutputErrors<<endl;};
	  void DirectGradientEstimation();
	  void BackProp();
	  void StepByGradient();
	  double GetOutputError(){ return sumOfOutputErrors;};
	  double TotalDatasetError(); // sum of errors for all rows of train data
	  void Train1();
	  void PrintOutputs();
      void DisplayResults();	
};

// loads inputsTraining and outputsTraining from "file"
int NN::LoadTrainingSet(string file,int nInp,int nHid,int nOut){
	std::ifstream data(file);
    std::string line;
    nTrainingEntries = 0;
    nInputs = nInp;
    nOutputs = nOut;
    nHiddenNeurons = nHid;
    // count number of lines in input file
    while(std::getline(data,line)) { nTrainingEntries++; }
    cout<<" There are "<<nTrainingEntries<<" entries in training dataset"<<endl;
    // reserve the memory
    inputsTraining = new double[nTrainingEntries*nInputs];
    outputsTraining = new double[nTrainingEntries*nOutputs];
    cout<<" Memory reserved..."<<endl;
    // rewind the file
    data.clear();
    data.seekg(0);
    // read training data file
    for(int iim = 0; iim<nTrainingEntries; iim++) {
		std::getline(data,line);
    	//cout<<" iim= "<<iim<<" Input: "<<line<<endl;
        std::stringstream lineStream(line);
        std::string cell;
        int count = 0;
        // break input string into inputs and answers
        while(std::getline(lineStream,cell,' ')) {
            //cout<<"count="<<count<<"cell="<<cell<<" "<<endl;
            if (count<nInputs) { 
				inputsTraining[iim*nInputs+count] = atof(cell.c_str()) ;
				//cout<<" count="<<count<<" Inp[][]="<<inputsTraining.GetElement(iim,count)<<endl;
			} else {
				outputsTraining[iim*nOutputs+count-nInputs] = atof(cell.c_str());
				//cout<<" count-nInputs="<<count-nInputs<<" Out[][]="<<outputsTraining.GetElement(iim,count-nInputs)<<endl;
			}	
          count++;
        } // while
        //cout<<" Input string "<<iim<<" parsed"<<endl;
        //char stop;
        //cin>>stop; 
    } // for
    cout<<" Training set loaded."<<endl;
    //char stop;
    //cin>>stop; 
    //inputsTraining.PrintMatrix();
    data.close();
	return 0;
}

int NN::LoadWorkingSet(string file,int nInp,int nHid,int nOut){
	std::ifstream data(file);
    std::string line;
    nTrainingEntries = 0;
    nInputs = nInp;
    nOutputs = nOut;
    nHiddenNeurons = nHid;
    // count number of lines in input file
    while(std::getline(data,line)) { nTrainingEntries++; }
    // reserve the memory
    inputsTraining = new double[nTrainingEntries*nInputs];
    // rewind the file
    data.clear();
    data.seekg(0);
    // read training data file
    for(int iim = 0; iim<nTrainingEntries; iim++) {
		std::getline(data,line);
    	//cout<<" iim= "<<iim<<" Input: "<<line<<endl;
        std::stringstream lineStream(line);
        std::string cell;
        int count = 0;
        // break input string into inputs and answers
        while(std::getline(lineStream,cell,' ')) {
            if (count<nInputs) { 
				inputsTraining[iim*nInputs+count] = atof(cell.c_str()) ;
			}
          count++;
        }
    }
    cout<<" Working set loaded."<<endl;
    data.close();
	return 0;
}


// reserves the memory and puts
//random values (range min-max) into weights  and biases
int NN::InitNet(double min, double max){
	
	cout<<" InitNet: nInputs="<<nInputs<<" nHiddenNeurons=";
	cout<<nHiddenNeurons<<" nOutputs="<<nOutputs<<endl; 
	// reserve the memory for weights and biases
	// hidden layer
	weightsHidden = new double[nHiddenNeurons*nInputs];
	biasHidden = new double[nHiddenNeurons];
	d_weightsHidden = new double[nHiddenNeurons*nInputs];
	d_biasHidden = new double[nHiddenNeurons];
	deltaHidden = new double[nHiddenNeurons];
	// output layer
	weightsOutput = new double[nHiddenNeurons*nOutputs];
	biasOutput = new double[nOutputs];
	d_weightsOutput = new double[nHiddenNeurons*nOutputs];
	d_biasOutput = new double[nOutputs];
	deltaOutput = new double[nOutputs];
	
	// current input and output vector, answers and error	
	currentInputs = new double[nInputs];
	currentOutputs = new double[nOutputs];
	currentAnswers = new double[nOutputs];
	currentError =  new double[nOutputs];
	
	// reserve memory for current net levels
	netHidden = new double[nHiddenNeurons];
	outHidden = new double[nHiddenNeurons];
	netOutput = new double[nOutputs];
	
	// make weights and biases random
	Randomize(weightsHidden,nHiddenNeurons*nInputs,min,max);
	Randomize(biasHidden,nHiddenNeurons,min,max);
	Randomize(weightsOutput,nHiddenNeurons*nOutputs,min,max);
	Randomize(biasOutput,nOutputs,min,max);
   
  	return 0;
}

// loads row of dataset into the net for estimation
void NN::GetTrainingEntry(int iTrainRow){
	for ( int i = 0 ; i<nInputs;i++)
	   currentInputs[i] = inputsTraining[iTrainRow*nInputs+i];
	for (int i = 0 ; i < nOutputs;i++)   
	  currentAnswers[i] = outputsTraining[iTrainRow*nOutputs+i];
}

// display digit on the screen as ASCII
void NN::DisplayDigit(int iImage){
  int scan = 0;
  for (int i = 0 ; i < 8; i++){
    for ( int j = 0 ; j < 8;j++){
       if (inputsTraining[iImage*nInputs + scan] > 0.0){
          cout<<"0";
       } else {
         cout<<"-";
       }
       scan++;
    }
    cout<<endl;
  }
}



// direct calculation of forward propagation
void NN::ForwardProp(){
	//  inputs ->  hidden layer
	// for each neuron in hidden layer
	for (int hid = 0 ; hid < nHiddenNeurons ; hid++){
		// combine inputs and add bias
        netHidden[hid] = biasHidden[hid]; 
        for (int inp = 0 ; inp < nInputs ; inp++){
//cout<<" hid="<<hid<<" inp="<<inp<<" ind="<<hid*nInputs + inp<<endl;			
		   netHidden[hid] = netHidden[hid] + 
		      currentInputs[inp]* weightsHidden[hid*nInputs + inp]; // b+w0*x0+w1*w0
	    }
	    outHidden[hid] = activation(netHidden[hid]); // y=sigma(b+w0*x0+w1*w0)
	}	
//int st;
//cin>>st;	
	sumOfOutputErrors = 0.0;
	// for each neuron in output layer
	for ( int out = 0 ; out < nOutputs ; out++){
		// combine hidden and add bias 
		netOutput[out] = biasOutput[out];
		for (int hid = 0 ; hid < nHiddenNeurons ; hid++){
			netOutput[out] = netOutput[out] +
			outHidden[hid]* weightsOutput[out*nHiddenNeurons+hid];
		}
		currentOutputs[out] = activation(netOutput[out]);
		currentError[out] = currentOutputs[out] - currentAnswers[out]; // e = y-t
		sumOfOutputErrors = sumOfOutputErrors + currentError[out]*currentError[out]; // e_total = e_total + e^2
	}
}

// calculate gradient by direct estimation
void NN::DirectGradientEstimation(){
	// calculate gradient for b, w0, w1;
	double e0;
	double e1;
	
	ForwardProp();
	e0 = sumOfOutputErrors;
	
	// hidden neurons bias calc
	for(int bHid=0; bHid < nHiddenNeurons; bHid++){		
		biasHidden[bHid] = biasHidden[bHid] + dw; // takes step for gradient estimation
		ForwardProp(); 
		e1 = sumOfOutputErrors; // error after step
		d_biasHidden[bHid] = (e1-e0)/dw; // d calc
		biasHidden[bHid] = biasHidden[bHid] - dw; // reverts step
	}
	
	// hidden neuron weight calc
	for(int wHid=0; wHid < nHiddenNeurons*nInputs; wHid++){
		weightsHidden[wHid] = weightsHidden[wHid] + dw;
		ForwardProp();
		e1 = sumOfOutputErrors;
		d_weightsHidden[wHid] = (e1-e0)/dw;
		weightsHidden[wHid] = weightsHidden[wHid] - dw;
	}
	
	// output neuron bias calc
	for(int bOut=0; bOut < nOutputs; bOut++){
		biasOutput[bOut] = biasOutput[bOut] + dw; // takes step for gradient estimation
		ForwardProp(); 
		e1 = sumOfOutputErrors; // error after step
		d_biasOutput[bOut] = (e1-e0)/dw; // d calc
		biasOutput[bOut] = biasOutput[bOut] - dw; // reverts step
	}
	
	// hidden neuron weight calc
	for(int wOut=0; wOut < nHiddenNeurons*nOutputs; wOut++){
		weightsOutput[wOut] = weightsOutput[wOut] + dw; 
		ForwardProp();
		e1 = sumOfOutputErrors;
		d_weightsOutput[wOut] = (e1-e0)/dw;
		weightsOutput[wOut] = weightsOutput[wOut] - dw;
	}
}


// calculate gradients by back-propagation
void NN::BackProp(){	
	//output layer delta calculation
	for(int out=0; out < nOutputs; out++){
		deltaOutput[out] = d_activation(netOutput[out])*currentError[out];
	}
	
	//hidden layer delta calculation
	for(int hid=0; hid < nHiddenNeurons; hid++){		
		deltaHidden[hid] = 0.0;
		for (int out = 0 ; out < nOutputs ; out++){
			deltaHidden[hid] = deltaHidden[hid] + deltaOutput[out]*weightsOutput[out*nHiddenNeurons+hid]*d_activation(netHidden[hid]);
		}
	}
	
	//assigning gradients for bias and weights in hidden layer
	for(int hid=0; hid < nHiddenNeurons; hid++){		
		double delta = deltaHidden[hid];
		d_biasHidden[hid] = delta; // bias gradient
		
		for(int inp=0; inp < nInputs; inp++){
			d_weightsHidden[hid*nInputs + inp] = delta*currentInputs[inp]; // weight gradient
		}	
	}
	
	//assigning gradients for bias and weights in hidden layer
	for(int out=0; out < nOutputs; out++){
		double delta = deltaOutput[out];
		d_biasOutput[out] = delta; // bias gradient
		
		for(int hid=0; hid < nHiddenNeurons; hid++){
			d_weightsOutput[out*nHiddenNeurons + hid] = delta*outHidden[hid]; // weight gradient
		}
	}
}

// change weights and biases in direction oppposite to gradient,
// scaled by learning rate (which should be negative)
void NN::StepByGradient(){
	for(int bHid=0; bHid < nHiddenNeurons; bHid++){		
		biasHidden[bHid] = biasHidden[bHid]+(learningRate*d_biasHidden[bHid]);
	}
	
	for(int wHid=0; wHid < nHiddenNeurons*nInputs; wHid++){
		weightsHidden[wHid] = weightsHidden[wHid]+(learningRate*d_weightsHidden[wHid]);
	}
	
	for(int bOut=0; bOut < nOutputs; bOut++){
		biasOutput[bOut] = biasOutput[bOut]+(learningRate*d_biasOutput[bOut]);
	}
	
	for(int wOut=0; wOut< nHiddenNeurons*nOutputs; wOut++){
		weightsOutput[wOut] = weightsOutput[wOut]+(learningRate*d_weightsOutput[wOut]);
	}
}

// calculates error for all entries in the dataset
// for current values of weights and biases
double NN::TotalDatasetError(){ // sum of errors for all rows of train data
//cout<<" There are "<<nTrainingEntries<<" rows in the dataset"<<endl;
	totalDatasetError = 0.0;
	for ( int entry = 0 ; entry < nTrainingEntries; entry++){
//cout<<"entry = "<<entry<<endl;
		GetTrainingEntry(entry);
	    ForwardProp();
	    totalDatasetError = totalDatasetError + GetOutputError();
	}
//cout<<" totalDatasetError/nEntries="<<totalDatasetError/nTrainingEntries<<endl;
	return totalDatasetError; //nTrainingEntries;
}


void NN::Train1(){
	// set net search parameters
	dw = 0.001;  // step to estimate gradient
	learningRate = -0.4;
    //DisplayDigit(iImage);
    int iImage = 0;
    srand (time(NULL));  // seed random number generator
    int searchStep = 0;
    
    while (( searchStep < 5000) && (TotalDatasetError() > 10.0) ){ 
  	  // pick random entry from training dataset
      iImage = nTrainingEntries*(double)rand() / RAND_MAX;
	  // copy inputs and outputs from training matrix into neural netg
  	  GetTrainingEntry(iImage);
      ForwardProp();
      //DirectGradientEstimation();
      BackProp();
      StepByGradient();
      cout<<"step: "<<searchStep;//<<" image: "<<iImage<<" Error for current row:"<<GetOutputError();
      cout<<" Total dataset error: "<< TotalDatasetError()<<endl;
      searchStep++;
    }
    cout<<" TRAINING COMPLETE"<<endl;
}

void NN::PrintOutputs(){
	cout<<" Net outputs: ";
	for (int out = 0 ; out < nOutputs ; out++){
		cout<<currentOutputs[out]<<"  ";
	}
	cout<<endl;
}


void NN::DisplayResults(){
	int iImage = -1;
	cout<<" There are "<< nTrainingEntries<<" entries "<<endl;
	
	while (iImage < nTrainingEntries) {
	  // copy inputs and outputs from big matrix
	  cout<<" Enter number of the entry to display: "; cin>>iImage;
	  GetTrainingEntry(iImage);
	  ForwardProp();
      DisplayDigit(iImage);
      //PrintVector(currentOutputs, nOutputs);
      PrintOutputs();
   }
    
}


int main(){
	NN neuralNet;
	neuralNet.LoadTrainingSet("train.txt",64,128,8);
	
	neuralNet.InitNet(-0.1,0.1);
	neuralNet.ForwardProp();
	//auto t7 = std::chrono::system_clock::now();
    //  neuralNet.DirectGradientEstimation(); 
	//auto t8 = chrono::system_clock::now();
	//cout<< " Timing Gradient:"<< chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() << " ms\n";
	
	neuralNet.Train1();
	neuralNet.LoadWorkingSet("work.txt", 64, 128, 8);
	neuralNet.DisplayResults();	
}
