from dotenv import load_dotenv
from transformers import pipeline
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import logging
import json
import os

class FeedbackClassifierPipeline:
    def __init__(self, labels, llm_model, groq_api_key, threshold=0.6, log_file="feedback_pipeline.log"):
        # -----------------------
        # Logging
        # -----------------------
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # -----------------------
        # Labels
        # -----------------------
        self.labels = labels
        self.threshold = threshold

        # -----------------------
        # Local Zero-Shot Classifier
        # -----------------------
        self.local_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )

        # -----------------------
        # LLM Fallback
        # -----------------------
        prompt_template = """
            You are a professional customer feedback handler. You receive hundreds of reviews each minute. Your task is:

            1. Classify the following review into one of the categories provided in {original_list}.
            2. If none of the existing categories fit, create a new category and use that.
            3. Consider the confidence scores in {confidence_list} to help decide if a new category is needed.

            ONLY return a single category as the answer.
            Return EXACTLY in the format: "Category: <category_name>"

            Feedback:
            {feedback}
            """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatGroq(model=llm_model, api_key=groq_api_key)
        self.fallback_chain = LLMChain(llm=llm, prompt=prompt)

    def _run_local_classifier(self, feedback):
        result = self.local_classifier(feedback, self.labels)
        top_label = result["labels"][0]
        top_conf = result["scores"][0]
        return top_label, top_conf, result

    def _run_llm_fallback(self, input_dict):
        result = self.fallback_chain.run(input_dict)
        final_result = result[len("Category:"):].strip() if result.lower().startswith("category:") else result.strip()
        return final_result

    def classify(self, feedback, confidence_list=None):
        if confidence_list is None:
            confidence_list = [0.0] * len(self.labels)

        input_dict = {
            "feedback": feedback,
            "original_list": self.labels,
            "confidence_list": confidence_list
        }

        # Run local classifier
        top_label, top_conf, local_result = self._run_local_classifier(feedback)
        self.logger.info(f"Feedback: {feedback}")
        self.logger.info(f"Local prediction: {top_label} (confidence: {top_conf:.2f})")

        # Decide on LLM fallback
        if top_conf >= self.threshold:
            final_category = top_label
            self.logger.info("Accepted local classifier prediction.")
        else:
            self.logger.info("Confidence below threshold, calling LLM fallback.")
            llm_result = self._run_llm_fallback(input_dict)
            final_category = llm_result
            self.logger.info(f"LLM fallback prediction: {final_category}")

        # Update categories dynamically if needed
        if final_category not in self.labels:
            self.labels.append(final_category)
            self.logger.info(f"New category added: {final_category}")

        # Log final output
        final_output = {
            "feedback": feedback,
            "final_category": final_category,
            "local_prediction": top_label,
            "confidence": top_conf
        }
        self.logger.info(json.dumps(final_output, indent=2))
        return final_output



if __name__=='__main__':
    load_dotenv('.env')
    labels = ["Bug Report", "Billing Issue", "Complaint", "Feature Request", "General Inquiry"]


    api_key = os.getenv('GROQ_API_KEY') 


    pipeline = FeedbackClassifierPipeline(
        labels=labels,
        llm_model='llama-3.1-8b-instant',
        groq_api_key=api_key,
        threshold=0.6,
        
    )


    feedback_text = "I really love the app, but I wish there was a built-in analytics dashboard showing my usage trends over time."
    result = pipeline.classify(feedback_text, confidence_list=[0.3, 0.6, 0.1, 0.0, 0.0])

    print(result)
