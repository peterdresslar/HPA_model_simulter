import streamlit as st

st.markdown(
    """
    ### Report on Assignment, CAS 522, Prof. Enrico Moriello 
    *Peter Dresslar*

    For this assignment, I have developed an extension to an existing Python app called HPA model simulter, which is available at
    https://github.com/AlonLabWIS/HPA_model_simulter. That code, written by Uri Alon lab member Yaniv Grosskopf, is a model representation of the
    hormonal production and regulation of the HPA (hypothalamic–pituitary–adrenal) axis as documented in (Milo et al. 2025).

    The original code, a model implemented within a Streamlit app, provides a simulation of the levels of: Corticotropin-releasing hormone (CRH), 
    Adrenocorticotropic hormone (ACTH), and of particular interest in the paper, Cortisol hormone over time using available biologically-appropriate parameters. 
    When run at the defaults using values from the Milo paper, the model demonstrates hourly behavior that appears biologically realistic.

    Upon initially reviewing the code, I noticed that when some random seeds were input into the noise parameter, the model would tend to suddenly destabilize 
    at the end of its run period. Seeing this, I suspected that the long term regulation of the gland systems, as described in the paper, would help
    stabilize the model if a second layer was added for gland mass dynamics. My extended model is called HPA model simulter fast-slow, and is in the root directory of
    this project. (The original model is in the `src` directory of the project.)

    To begin, I added an entropy widget to user interface so that the user can control random seeding of the model runs, thus allowing reproducibility. 
    In order to compare outputs of the newly-updated and base models, I have added this capability to the prior version of the model in `\src` also. 

    To add in a model layer for gland mass growth / reduction effects, I took information both from the Milo paper and the supplied source code notebook from
    that paper's supplementary data (https://github.com/tomermilo/hpa-drugs). Where there were differences between the notebook and the paper, I generally went
    with the paper's calculations, though I did import a modulatory for gland mass growth from the source notebook. The new gland layer parameters added to the left hand 
    user-interface of the app also default to the paper's values.

    An additional plot was added for the new layer. A toggle has been added to the user interface to turn the glandular layer function on or off, and a slider is available
    that allows the user to tweak the relative synchonization of the two layers.

    The updated model is available here:  
    https://hpamodel-sim-fast-slow.streamlit.app/

    ### Observations / Results
    Running the new model, it is clear to see the real-world stabilizing function that the glands provide to hormone system regulation. The fluctuations of the
    model without glands are smoothed dramatically, as we would expect and as is reported by the literature. The glandular system generally moves over days and weeks where
    the hormone level fluctuate daily, as we would also expect. We can also observe the reaction of relatively fast gland growth and slow reversion to normal over 
    a period of weeks in response to sharp stressor episodes. 

    More generally, the model now illustrates the nature of general biological regulatory capacity. Many systems of biology operate on this slower timescale, essentially
    soaking up the fluctuations of immediate stimuli to stabilize the organism over time. It is interesting to consider the long-term evolutionary fitness of what is
    effectively a quantitative storage relationship between these fast and slow systems.

    ### Future work
    As a result of this work, I have communicated about the updates to the model to the Uri Alon lab, and suggested a pull request with updates to 
    both versions of the model.

    In the future, it would be possible to extend this model further with more detailed glandular activity representing factors such as functional saturation in
    over-grown gland masses. It would be possible also to add somatic cycle effects on hormone levels as biologically appropriate.

    """
)