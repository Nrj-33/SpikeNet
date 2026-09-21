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
