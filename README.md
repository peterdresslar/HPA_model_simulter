# HPA_model_simulter
A Streamlit app simulating the Hypothalamic-Pituitary-Adrenal (HPA) axis using stochastic differential equations.

The goal for this repository is to extend existing code to emphasize the multi-timescale nature of the hypothalamus-pituitary-adrenal (HPA) axis's biological activity. The complex exhibits a fast-slow (or plant-controller) relationship in responses to immediate stimuli which regulate medium and long term outcomes to those stimuli. While these control mechanisms are likely bioprotective and evolutionarily advantageous, they also can present challenges for therapeutic intervention in many situations.
 
To explore the relationship, we will extend an example developed by the Uri Alon Laboratory.

The original code in this repo was created by Yaniv Grosskopf, Uri Alon Lab, Weizmann Institute of Science. 
https://github.com/AlonLabWIS/HPA_model_simulter

It was then extended using observations from the paper, Milo, T., Nir Halber, S., Raz, M., Danan, D., Mayo, A., & Alon, U. (2025). Hormone circuit explains why most HPA drugs fail for mood disorders and predicts the few that work. Molecular Systems Biology, 21(3), 254–273. https://doi.org/10.1038/s44320-024-00083-0. We have reviewed functions of the HPA_full model from the sample code at the location specified in that paper: https://github.com/tomermilo/hpa-drugs

Our model is called HPA Model Simulter Fast Slow. It was created by Peter Dresslar as part of the Arizona State University Complex Systems class taught by Prof. Enrico Borriello.

To run locally `uv run streamlit run HPA_model_simulter_fast_slow.py`.
