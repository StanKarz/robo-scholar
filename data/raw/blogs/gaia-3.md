2 December 2025 | Research

# GAIA-3: Scaling World Models to Power Safety and Evaluation

Transforming world modeling from a tool for visual synthesis into a foundation for autonomy evaluation.

Evaluating autonomous-driving systems at scale remains one of the defining challenges in advancing real-world autonomy. Real-world testing is essential for proving safety, but it is costly, logistically constrained, and increasingly data-inefficient. As driving models improve and make fewer observable mistakes, the number of miles needed for statistically meaningful conclusions rises sharply. Most of those miles are uneventful, offering little signal about rare, safety-critical behavior.

Simulation provides a path forward. Virtual environments can enable safe, reliable, repeatable, and scalable testing of driving models. Yet, despite their promise, existing simulation approaches fall short of enabling meaningful evaluation for modern end-to-end driving systems. Procedural simulators, the long-standing standard for autonomy testing, allow precise control but lack realism. 3D reconstruction-based simulators achieve greater realism but struggle with occlusions and dynamic agents. **Our latest advances in world models deliver the best of both worlds: **capturing the statics and dynamics of real environments while generating realistic counterfactuals that expand the coverage of real-world testing.

*GAIA-3 is conditioned to weave around a highway scene in a sine wave, while other agents’ trajectories are un-modified.*

*GAIA-3 is used to create a safety-critical scenario by altering the ego vehicle’s trajectory to create a collision with an oncoming truck whose trajectory is unmodified from the original real video.*

*GAIA-3 generates three diverse variants of a seed scene, observing the exact same scenario in sunset, bright sun, and night. GAIA-3 can not only generate a scene with controllable lighting, but it can also render a precise scene with realistic and controllable weather and lighting conditions.*

*GAIA-3 is used to transfer a scene to three distinct embodiments with different camera positions. The underlying scenario is untouched, but rendered with fidelity as if it had been observed from 3 different vehicles.*

Wayve has been pioneering the use of world models to unlock a new paradigm in how autonomous vehicles are trained and evaluated. Building on our work in future prediction, dreaming about driving, predicting in a bird’s eye view, and learning a world model, we introduced **GAIA-1** (Generative Artificial Intelligence for Autonomy) as our first step toward this vision. GAIA-1 demonstrated that generative models could learn from video, text, and action to produce realistic driving experiences. **GAIA-2** expanded the horizon by introducing richer controllability, broader geographic coverage, and diverse vehicle embodiments through multi-camera, spatio-temporally coherent scene generation.

With **GAIA-3**, we take a bold step forward: transforming world modeling from a tool for visual synthesis into a foundation for autonomy evaluation. GAIA-3 generates driving scenes that are not only lifelike, but also structured and purposeful—designed to measure, compare, and accelerate progress toward safe, scalable autonomous driving.

GAIA-3 combines the realism of real-world data with the control of simulation. It allows us to take authentic driving sequences and re-drive them with precise, parameterized variations—for example, altering the ego vehicle’s trajectory while every other element in the scene remains perfectly consistent. Other agents keep their motion, lighting, and weather unchanged, and the world stays coherent. This ability to operate in a world-on-rails approach is a major step forward for generative world modeling, transforming evaluation from reactive measurement to proactive exploration of the edge cases that define safety.

The following example shows a continuous highway sequence generated with a programmatically controlled sine-wave camera motion, demonstrating GAIA-3’s ability to alter the ego vehicles’ trajectory, while maintaining all other scene elements coherently over an extended period of time.

*GAIA-3 weaves around a coastal road in a sine wave. Note the realistic brake lights of the other agents, and the detail of the cyclist on the right as we pass.*

*GAIA-3 weaves around a suburban road, stuck in traffic. Even as we weave into unrealistic positions, the fine-grained details of other actors remain clear, as well as the buildings and flags on our right.*

*GAIA-3 weaves around a single lane road. Note the realistic foliage on the sides and the realistic shadows as the car cruises under some trees.*

## Safety and Evaluation with GAIA-3

Safety and evaluation are central to developing robust autonomous systems. Building trust in a driving model’s reliability requires not only strong on-road performance, but also the ability to measure, reproduce, and stress-test its behaviour offline.

GAIA-3 supports this by providing a suite of highly controllable and consistent generation tools that enable structured testing under realistic conditions. Together, these capabilities transform world-model generation into a platform for systematic evaluation, one that supports repeatable experiments and measurable comparisons. Each capability targets a key limitation in current industry testing workflows, directly linking data generation to measurable outcomes.

### Safety-critical scenarios

Real-world safety-critical events—collisions, near-misses, or loss-of-control situations—are rare, unpredictable, and far too dangerous to recreate intentionally. Today, the industry still leans on controlled test-track experiments, such as NCAP (New Car Assessment Program) tests, with staged actors and dummy vehicles. These setups offer precision but come at the cost of realism and scalability. They cannot capture the visual richness, behavioral diversity, and environmental complexity of real-world driving. As a result, test-track experiments can only ever form part of the solution and risk creating an incomplete picture of real-world safety performance.

GAIA-3 overcomes these limitations by enabling controlled, realistic variations of real-world driving sequences. Given an existing scene, GAIA-3 can alter the ego vehicle’s trajectory while keeping the rest of the environment consistent: other agents continue their original motion, and static elements remain unchanged. This makes it possible to generate counterfactual yet realistic safety-critical scenarios, such as the ego vehicle drifting into oncoming traffic or other road users, like a cyclist or vehicle speeding up, as shown in the examples below.

*GAIA-3 is conditioned to veer out of the current driving lane and collide with a car moving in the opposite direction, generating a safety-critical incident.*

*GAIA-3 is conditioned to accelerate and collide head on with another vehicle waiting to turn, generating a safety-critical situation.*

*GAIA-3 is conditioned to drift out of the lane, colliding with the side of a vehicle in the other lane.*

The same methodology can be used to generate NCAP-style tests virtually and at scale, both within simulated test-track environments and across varied real-world conditions. While vehicle trajectories and timing remain consistent, the background, lighting, and scene dynamics change.

This capability unlocks systematic generation of crash and near-crash scenarios that can be evaluated using the same occupancy and trajectory metrics applied to real-world data—paving the way for scalable, reproducible safety validation.

*GAIA-3 can generate augmentations to produce data for NCAP testing scenarios such as Car-to-Car front turn-across-path (CCFTAP). In this video, GAIA-3 generates a high-speed collision by driving across the path of oncoming traffic on a test track.*

*GAIA-3 generates a high-speed collision by driving across the path of oncoming traffic in an urban scenario, mirroring the scene produced in the CCFTAP Test Track scenario. This video can be used as a Car-to-Car front turn-across-path (CCFTAP) testing scenario.*

*GAIA-3 can generate augmentations to produce data for NCAP testing scenarios such as Car-to-Car rear-stationary (CCRS). In this video, GAIA-3 generates a video of the ego vehicle moving forward to strike the rear of another, stationary vehicle, on a test track.*

*GAIA-3 generates a collision by striking the rear of another, stationary vehicle in an urban scenario, mirroring the scene produced in the CCRS Test Track scenario. This video can be used as a Car-to-Car rear-stationary (CCRS) testing scenario.*

A crucial benchmark for safety-critical generation is consistency: ensuring that when only the ego vehicle’s behavior changes, the rest of the scene remains physically and visually coherent. To validate this, we use real-world sequences captured with LiDAR and modify the ego vehicle’s trajectory to produce collisions with scene objects. We then align the LiDAR point clouds from the original recordings with the generated frames and check for misalignments and inconsistencies. Below, we showcase an example confirming that GAIA-3 preserves spatial structure and realism even when synthesizing safety-critical scenarios.

Consistency comparison:

*Left: An original real driving scene with safe behaviour, and LiDAR detections overlaid.*

*Right: A GAIA-3 generation, driving the ego vehicle into an oncoming vehicle while the generation remains aligned with the translated LiDAR scans.*

*Left: An original real driving scene with safe behaviour, and LiDAR detections overlaid.*

*Right: A GAIA-3 generation, creeping the ego vehicle forward into the lead vehicle, while the transformed lidar scans and structure are preserved.*

*Left: An original real driving scene with safe behaviour, and LiDAR detections overlaid.*

*Right: A GAIA-3 generation, accelerating the ego vehicle into the lead cyclist, while the transformed LiDAR structure is preserved.*

### Offline Evaluation Suites

Real-world driving rarely follows a script. Unexpected events, such as an abrupt stop, a late merge, or a pedestrian stepping into the road, reveal how well a model truly understands its environment. The ability to recreate and test these “what-if” moments offline is essential for building confidence in autonomous systems. With GAIA-3, this can be done systematically.

By conditioning the ego vehicle’s behavior through** action conditioning **and optionally combining it with World-on-Rails perturbations, GAIA-3 can generate controlled variations of real-world scenarios. From a single recorded sequence, an entire family of “what-ifs” can be created by adjusting different parameters. These perturbations allow quantitative testing of a model’s ability to recover from edge cases or maintain stability under changing conditions.

The result is structured offline evaluation test suites that are scalable, repeatable, and measurable. They provide richer diagnostic signals than static replay, revealing how policy behavior shifts when conditions change. Correlation studies between GAIA-3’s synthetic interventions and on-road experiments indicate that the model can reliably predict relative policy performance, increasing the practical value of offline evaluation for model comparison and decision-making.

The examples below illustrate this: GAIA-3 is conditioned to drift to the left/right or go too fast/slow (trajectory in green), while the driving model should predict to bring the vehicle back to a safe position (trajectory in pink).

In the video above, GAIA-3 is used to create multiple four-example test cases of sub-optimal behaviour across five different scenes. In each, GAIA-3 drives to the left, right, too fast, and too slow. We see the worlds remain consistent, “on-rails,” providing useful counterfactual intervention states to evaluate a driving model’s ability to recover.

### Embodiment Transfer

Different camera rigs and fields of view make it challenging to reuse data across vehicles. With **embodiment transfer,** GAIA-3 can re-render the same scene from a new sensor configuration, using only a small, *unpaired* sample from the target rig.

This means evaluation suites can be easily transferred across different embodiments or automotive OEM vehicle programs without requiring paired captures. The examples below show how GAIA-3 transfers a scene from one camera rig to another.

**Original Embodiment:** The original videos are captured from a vehicle fitted with a rig of 5 RGB cameras, one looking straight ahead, two looking towards the sides, and two more looking backwards.

**Embodiment A:** This vehicle is a different make and model from the Original Embodiment, but with a similar 5-camera configuration. Key differences are the panel that obscures the front forward camera view, the vehicle’s bonnet that appears flatter and shorter in the left and right forward cameras, and a more visible windscreen. GAIA-3 convincingly reproduces reflections from the scene, including those from oncoming vehicles.

**Embodiment B:** This is a similar vehicle to the Original Embodiment, but with a different camera rig. The front forward camera has a wider, more zoomed-out fisheye lens. The left and right forward cameras are angled further outward, and the rear cameras are angled further back and zoomed in.

Below are several examples demonstrating the consistency and realism achieved through this methodology.

*GAIA-3 is used to transfer a scene from the USA to three distinct embodiments. Note the precise transfer of the lead vehicle’s brake lights and blinkers, preserved as we observe the scene from different embodiments.*

*GAIA-3 is used to transfer a scene from Japan to three distinct embodiments. Note the high-frequency details in the foliage, realistically rendered as the scene is rendered from different embodiments.*

*GAIA-3 is used to transfer a scene from Germany to three distinct embodiments. Note the cyclist’s fidelity, as we observe them from diverse camera setups in the generated renders.*

*GAIA-3 is used to transfer a scene from the UK to three distinct embodiments. Note the realistic lighting changes of the cameras, even as they shift with different embodiments.*

### Robustness and Interpretable Control

Driving models must remain reliable even when faced with changes in appearance, lighting, or semantics. However, these variations must be measurable to ensure meaningful evaluation.

We introduce controlled visual diversity, allowing the appearance of a scene to change while its underlying structure remains the same. This means elements like lighting, textures, and weather can vary, but the geometry and motion of the scene stay consistent. As a result, we can directly compare model performance across different visual conditions, evaluate robustness at scale, and better understand how specific visual changes affect model behavior.

*GAIA-3 is used to augment a scene appearance preserving scenario’s semantics.*

*GAIA3 is used to augment a scene appearance preserving scenario’s semantics.*

*GAIA3 is used to augment a scene appearance preserving scenario’s semantics.*

*GAIA3 is used to augment a scene appearance preserving scenario’s semantics.*

### Enrichment and Debugging

Rare failure modes are difficult to capture in real-world driving, limiting data coverage and slowing model iteration. GAIA-3 can learn from a small number of examples and generate structured variations around them, expanding scenario families such as braking or merging into rich, physically consistent test sets.

These rare events can be amplified into larger, labeled suites for focused testing or retraining, reducing the time between discovering a problem and verifying a fix.

The example below shows a specific behavior, harsh braking, generated as controlled out-of-distribution variations that help evaluate model behavior in situations that are difficult to reproduce through real-world testing. The same behavior can be translated across different environments and countries. For example, braking on a highway in the US, in an urban setting in Japan, or even stopping unexpectedly at a green light.

*GAIA-3 can help expand datasets with rare out-of-distribution examples. Here, GAIA-3 can translate a specific failure mode—harsh braking in the middle of the street—into new scenes and geographies.*

## Improved Capabilities through Scale

Progress in world modeling is driven by scale: not only in parameters, but in data diversity, representational power, and the quality of generated experience.

GAIA-3 is a **15-billion-parameter latent diffusion-based world model **designed for scalable, realistic, and controllable offline evaluations for autonomous driving. To support this capacity, GAIA-3 was trained using five times more compute than GAIA-2 and on roughly ten times more data, spanning 9 countries across 3 continents. 

The dataset emphasizes safety-critical scene elements such as pedestrians, cyclists, signs, and traffic control infrastructure, ensuring that the model learns not just to imitate driving scenes but to understand and reproduce the elements most relevant to autonomous systems.

GAIA-3 yields clear improvements with sharper visuals of static and dynamic scene elements. Importantly, it also shows enhanced world modelling capabilities that maintain scene coherence over long trajectories and through moments of temporary occlusion. This example demonstrates the quality and consistency of the pedestrian in the rear camera through occlusion.

The following examples showcase GAIA-3’s ability to generate realistic pedestrians in motion, text signs that are clear and readable, important landmarks being generated faithfully and fine-grained details being preserved.

*GAIA-3 is able to generate realistic pedestrian motion in this fairly busy rainy scene.*

*GAIA-3 is able to render text onto street signs and onto the road.*

*GAIA-3 faithfully renders the Arc de Triomphe in Paris as the ego car drives towards it.*

*GAIA-3 faithfully generates pedestrians, bicyclists and motorbikes as the ego car approaches the Big Ben in London.*

## Comparison between GAIA-2 and GAIA-3

GAIA-3 represents a significant leap forward in scale and capability. It doubles the model size compared to its predecessor, GAIA-2, greatly expanding representational capacity and generative precision. This broader foundation enables GAIA-3 to generalize across **geographies, embodiments, and driving contexts, **making it a truly global model for autonomy evaluation. 

At the heart of this scale-up is a **new video tokenizer **twice the size of GAIA-2’s. It captures safety-critical spatial and temporal structures, from subtle pedestrian motion to fast-moving vehicles, road signs, and traffic lights. By encoding fine-grained spatio-temporal context, GAIA-3 represents the physical and causal structure of real-world driving more faithfully than ever before. The model produces higher-fidelity video generations with sharper visuals, more consistent lighting, and richer texture detail. 

The following examples illustrate GAIA-3’s notable improvements over GAIA-2, particularly in the rendering of key traffic elements, including signage.

## The Road Ahead

GAIA-3 refines world modeling into a practical framework for evaluation and validation by enabling controllability, enhanced realism, and utility in a single system.

The result is a model that supports structured, repeatable testing, which is an important step toward scalable evaluation of end-to-end driving systems. The capabilities provide a grounded way to evaluate progress and compare models offline using metrics that reflect real-world performance.

We continue to focus on efficiency and real-time generation, as well as the validation of the tool through our UK government grant-funded project, DriveSafeSim. Our goal is to establish generative simulation as the primary tool for measuring progress and proving safety across the field of embodied AI.