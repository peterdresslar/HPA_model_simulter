# HPA_model_simulter
A Streamlit app simulating the Hypothalamic-Pituitary-Adrenal (HPA) axis using stochastic differential equations.

The goal for this repository is to extend existing code to emphasize the multi-timescale nature of the hypothalamus-pituitary-adrenal (HPA) axis's biological activity. The complex exhibits a fast-slow (or plant-controller) relationship in responses to immediate stimuli which regulate medium and long term outcomes to those stimuli. While these control mechanisms are likely bioprotective and evolutionarily advantageous, they also can present challenges for therapeutic intervention in many situations.
 
To explore the relationship, we will extend an example developed by the Uri Alon Laboratory. In particular, we add functions and parameterization for glandular (pituitary and adrenal) development and regulation, and process the hormonal and glandular functions togehter as a dual-layer dynamic system. We also add a synchronizing control: while the app defaults to the values and timescales proposed in the paper, the user is able to adjust the relative speed of the glandular adaptation layer to see the resultant effects on the overall system.

One small additional change is the ability to control random seeding (entropy) of the noise parameter, for the purpose of reproducibility.

<p align="center">. . .</p>

The original code in this repo was created by **Yaniv Grosskopf**, Uri Alon Lab, Weizmann Institute of Science. 
https://github.com/AlonLabWIS/HPA_model_simulter

It was then extended using observations from the paper, Milo, T., Nir Halber, S., Raz, M., Danan, D., Mayo, A., & Alon, U. (2025). Hormone circuit explains why most HPA drugs fail for mood disorders and predicts the few that work. Molecular Systems Biology, 21(3), 254–273. https://doi.org/10.1038/s44320-024-00083-0. We have reviewed functions of the HPA_full model from the sample code at the location specified in that paper: https://github.com/tomermilo/hpa-drugs

Our model is called `HPA_Model_Simulter_Fast_Slow.` It was created by Peter Dresslar as part of a Arizona State University Complex Systems class taught by Prof. Enrico Borriello.

To run locally: `uv run streamlit run HPA_model_simulter_fast_slow.py`.
