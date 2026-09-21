//=========================================================================================================================
//Interactive Spiking Neuron Network-SpikeNet
//----------------------------------------------------------------------------------------------------------------------
//10 Input Neuron -> 5 Output Neurons(50Synapses), all LIF neurons.
//Combine 5 self learning mechanisms all running automtically on 
//every pattern you type in 
//1. LIF Neuron Dynamics             (base spiking behaviour)
//2. Adaptive Threshold              (homeostasis/ self stabilization)
//3. Lateral Inhibition              (competition based outputs)
//4. Reward modulated STDP (R-STDP)    (learns from right or wrong feedback)
//5. Weight normalization            (keeps weights from saturating)

#include<iostream>
#include<sstream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<iomanip>

using namespace std;

const int INPUT_SIZE    =10;
const int OUTPUT_SIZE   =5;
const int TIME_STEP     =50;
const double DT     = 1.0;
const double LEARNING_RATE      =1.0;
const double INHIBITION_STRENGTH     =1.3;//raised 10 inputs need stronger competition
const double THRESHOLD_INCREMENT     =0.15;//how much threshold can raise per spike
const double THRESHOLD_DECAY_TAU     =30.0;//time constant for threshold decay
const double TARGET_WEIGHT_SUM          =1.6;//lowered: stops combined current from saturating output neurons

//=========================================================================================================================
//LIF Adaptive(homeostatic) threshold
//=========================================================================================================================
class LIFNeuron
{
    public:
    double V=0.0; //membrane potential
    double V_rest=0.0;
    double V_reset=0.0;
    double base_threshold=1.0;  
    double tau = 10.0; //membrane time constant
    int refractory_period = 0; //refractory period in time steps
    int refractory_timer = 0; //timer for refractory period
    double threshold = base_threshold;

    void reset()
    {
        V=V_reset;
        refractory_timer = 0;
    //NOTE:threshold is intentionally not reset-> it persists across trials which makes homeostasis meaningful over time
    }

    bool step(double I)
    {
        //Threshold relaxes back every timestep
        threshold +=    ( base_threshold - threshold ) / THRESHOLD_DECAY_TAU;
        
        if (refractory_timer > 0)
        {
            --refractory_timer;
            V = V_rest; 
            return false;
        }
        V += ( - (V - V_rest) + I ) / tau * DT;
        if (V>= threshold)
        {
            V=V_reset;
            refractory_timer= refractory_period;
            threshold += THRESHOLD_INCREMENT;//fire-> harder to fire again soon
            return true;
        }
        return false;
    
    }
    void inhibit(double amount)
    {
        V -= amount;
        if(V< V_rest-1.0)
        {
            V = V_rest - 1.0;
        }
    };
};

    //-------------------------------------------------
    //Synapse with eligibility trace for R-STDP
    //-------------------------------------------------
    class Synapse
    {
        public:
        double weight;
        double eligibility;
        double A_plus=0.05;
        double A_minus=0.05;
        double tau_stdp=20.0;

        Synapse(double w) : weight(w) {}

void accumulationEligibility(int dt_spike)
{
if(dt_spike>0)
{
    eligibility +=A_plus *std::exp(-dt_spike/tau_stdp);
}
else if(dt_spike<0)
{
    eligibility -=A_minus *std::exp(dt_spike/tau_stdp);
}
}
void applyReward(double reward)
{
    weight += LEARNING_RATE * reward * eligibility;
    if(weight<0.0)
    {
        weight=0.0;
}
if(weight>3.0)
{
    weight=3.0;
}
eligibility=0.0;
}
};

double encodeInput(int pixel)
{
    return pixel == 1 ? 1.2 :0.05;
}

//Rescale incoming neuron's weight so they sum fixed budget.
// This stops connections fromm saturating to the max together

void normalizeIncomingWeights(vector<vector<Synapse>>& synapses, int o)
{
double sum=0.0;
for (int i=0;i<INPUT_SIZE;i++)
{
    sum+=synapses[i][o].weight;
}
if (sum > 1e-6)
{
    double scale = TARGET_WEIGHT_SUM / sum;
    for (int i=0; i<INPUT_SIZE; i++)
        synapses[i][o].weight *= scale;
}
}

void printWeights(std::vector<std::vector<Synapse>>& synapses)
{
    std::cout<<"\n Current synaptic weights (rows=input, cols=output)---\n";
    for (int o = 0; o< OUTPUT_SIZE; o++)
    {
        std::cout<<"Out"<<o<<"  ";
        std::cout<<"\n";
        for (int i = 0; i < INPUT_SIZE; i++)
    {
        std::cout<<std::fixed << std::setprecision(2)<<std::setw(6)<<synapses[i][o].weight << " ";

    }
    std::cout<<"\n";
    }
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));\
    std::vector<LIFNeuron> inputNeurons(INPUT_SIZE);
    std::vector<LIFNeuron> outputNeurons(OUTPUT_SIZE);
    std::vector<std::vector<Synapse>> synapses(INPUT_SIZE, std::vector<Synapse>(OUTPUT_SIZE, Synapse(0.3)));

    //Break Symmetry so output neurons don't learn identically
    for (int i=0; i<INPUT_SIZE; i++)
    {
        for(int o = 0; o< OUTPUT_SIZE;o++)
        {
            synapses[i][o].weight = 0.2 + 0.2 * (static_cast<double>(rand()) /  RAND_MAX);
        }
    }

    std::cout<<"\n ===================================================";
    std::cout<<"\n SpikeNet: Interactive Spiking Neuron Network Model";
    std::cout<<" "<<INPUT_SIZE<<" Input Neurons" <<OUTPUT_SIZE<<" Output Neurons";
    std::cout<<"\n ===================================================\n";
    std::cout<<"Enter a pattern as "<<INPUT_SIZE<<" spaced 0s and 1s (e.g., 1 0 1 0 1 0 1 0 1 0) or type 'exit' to quit:\n";
    std::cout<<" Type q instead of a pattern at any time to quit the program\n";
    int trial = 0;
    int correctCount = 0;

    while (true)
    {
        std::cout<<"\n [Trial "<<trial<<"] Enter pattern ("<<INPUT_SIZE<<" values, e.g. 0 1 0 1 0 1 0 1 0 1):";
        std::string line;
        std::getline(std::cin, line);
        if(line == "q" || line == "Q")
        {
            break;
        }
        bool validInput = true;
        std::istringstream iss(line);
        std::vector<int> pattern(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; ++i)
        
        {
            if(!(iss >> pattern[i]) ||(pattern[i] != 0 && pattern[i] != 1))
            {
                validInput = false;
                break;
            }
        }

        if (!validInput)
        {
            std::cout<<"Invalid input. Please enter "<<INPUT_SIZE<<" values of 0 or 1 separated by spaces.\n";
            continue;
        }

    std::cin.ignore(1000,'\n');//clear trailing newline before getting next getline.
    //-- Reset neuron membrane state for this trial (threshold persists)-----
    for (auto& n : inputNeurons) n.reset();
    for (auto& n : outputNeurons) n.reset();

    std::vector<int> lastInputSpikeTimes(INPUT_SIZE, -1000);
    std::vector<int> lastOutputSpikeTimes(OUTPUT_SIZE, -1000);
    std::vector<int> SpikeCounts(OUTPUT_SIZE, 0);
    std::vector<bool> prevInputSpiked(INPUT_SIZE, false);

    for(int t =0; t<TIME_STEP; t++)
    {
        //INPUT LAYER
        std::vector<bool> currentInputSpikes(INPUT_SIZE, false);
        for(int i=0; i<INPUT_SIZE; i++)
        {
            currentInputSpikes[i] = inputNeurons[i].step(encodeInput(pattern[i]));
            if(currentInputSpikes[i])
            {
                lastInputSpikeTimes[i] = t;
            }
        }
        // OUTPUT LAYER
        std::vector<bool> currentOutputSpikes(OUTPUT_SIZE, false);
        for (int o = 0; o < OUTPUT_SIZE; o++)
        {
            double I_out = 0.0;
            for (int i = 0; i < INPUT_SIZE; i++)
                if (prevInputSpiked[i]) I_out += synapses[i][o].weight;
            currentOutputSpikes[o] = outputNeurons[o].step(I_out);
            if (currentOutputSpikes[o])
            {
                lastOutputSpikeTimes[o] = t;
                SpikeCounts[o]++;
            }
        }

        // Lateral inhibition
        for (int o = 0; o < OUTPUT_SIZE; o++)
            if (currentOutputSpikes[o])
                for (int o2 = 0; o2 < OUTPUT_SIZE; o2++)
                    if (o2 != o) outputNeurons[o2].inhibit(INHIBITION_STRENGTH);

        prevInputSpiked = currentInputSpikes;
    }
 // Decide the winner
        int winner = -1;
        int maxCount = -1;
        bool tie = false;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            if (SpikeCounts[o] > maxCount) { maxCount = SpikeCounts[o]; winner = o; tie = false; }
            else if (SpikeCounts[o] == maxCount && maxCount > 0) { tie = true; }
        }
        if (maxCount <= 0) winner = -1;
        if (tie) winner = -1;
 
        int target = pattern[0] % OUTPUT_SIZE;
        double reward = (winner == target) ? 1.0 : (winner == -1 ? -0.2 : -1.0);
        if (winner == target) correctCount++;
 
        // Apply reward-modulated STDP, then normalize weights per output neuron
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            for (int i = 0; i < INPUT_SIZE; i++)
                synapses[i][o].applyReward(reward);
            normalizeIncomingWeights(synapses, o);
        }
 
        // --- Report this trial ---
        std::cout << "  Spike counts: ";
        for (int o = 0; o < OUTPUT_SIZE; o++) std::cout << "Out" << o << "=" << SpikeCounts[o] << " ";
        std::cout << "\n  Network's answer: "
                  << (winner == -1 ? std::string("no clear winner") : ("Output " + std::to_string(winner)))
                  << " | Target: " << target
                  << " | Reward: " << reward
                  << (winner == target ? "  [CORRECT]" : "  [INCORRECT]") << "\n";
 
        trial++;
        std::cout << "  Running accuracy: " << correctCount << "/" << trial
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * correctCount / trial) << "% )\n";
    }
 
    printWeights(synapses);
    std::cout << "\nSession ended after " << trial << " trials. Goodbye.\n";
    return 0;
}
