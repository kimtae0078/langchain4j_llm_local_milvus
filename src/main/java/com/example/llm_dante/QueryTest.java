package com.example.llm_dante;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import io.milvus.client.MilvusServiceClient;
import io.milvus.common.clientenum.ConsistencyLevelEnum;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;

import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;


public class QueryTest {
    public static void main(String[] args) {

        String question = "미사일";

        EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        Embedding questionEmbedding = embeddingModel.embed(question).content();

        EmbeddingStore<TextSegment> embeddingStore = MilvusEmbeddingStore.builder()
                .host("localhost")
                .port(19530)
                .collectionName(CONSTANT.COLLECTION_NAME)
                .dimension(384)
                .indexType(IndexType.FLAT)
                .metricType(MetricType.L2)
                .build();

        /*
        // ***** milvusClient Search S *****
        final MilvusServiceClient milvusClient = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost("localhost")
                        .withPort(19530)
                        .build()
        );

        // 컬렉션을 메모리에 로드
        milvusClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName(CONSTANT.COLLECTION_NAME)
                        .build()
        );

        int maxResults = 10;
        List<Float> vectors = questionEmbedding.vectorAsList(); // question vector 값 list

        System.out.println("\n\n\n\n\n============================question============================\n\n");
        System.out.println(Collections.singletonList(vectors).get(0));
        System.out.println("\n\n============================question============================\n\n\n\n\n");


        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(CONSTANT.COLLECTION_NAME)
                .withConsistencyLevel(ConsistencyLevelEnum.STRONG)      // 일관성 수준(STRONG, BOUNDED, EVENTUALLY)
                .withMetricType(MetricType.L2)
                .withVectors(Collections.singletonList(vectors))        // 질의문
                .withOutFields(Arrays.asList(CONSTANT.PK_FIELD, CONSTANT.TEXT_FIELD))    // 반환할 필드들
                .withTopK(maxResults)                                   // 반환할 답 최대 수
                .withVectorFieldName(CONSTANT.VECTOR_FIELD)             // 벡터 필드명
                .build();

        R<SearchResults> respSearch = milvusClient.search(searchParam);

        SearchResultsWrapper wrapperSearch = new SearchResultsWrapper(respSearch.getData().getResults());

        System.out.println("\n\n\n\n\n============================result S============================\n");
        System.out.println(wrapperSearch.getIDScore(0));
        System.out.println("\n============================result E============================\n\n\n\n\n");
        //System.out.println(wrapperSearch.getFieldData(CONSTANT.PK_FIELD, 0));
        // ***** milvusClient Search E *****

        */

        int maxResults = 3;
        double minScore = 0.5;

        System.out.println("\n\n\n\n\n============================question============================\n");
        System.out.println(questionEmbedding.toString());
        System.out.println("\n============================question============================\n\n\n\n\n");

        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.findRelevant(questionEmbedding, maxResults, minScore);

        String information = relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(Collectors.joining("\n\n"));

        System.out.println("\n============================relevantEmbeddings S============================\n");
        for (EmbeddingMatch<TextSegment> relevantEmbedding : relevantEmbeddings) {
            System.out.println(relevantEmbedding.toString());
        }
        System.out.println("\n============================relevantEmbeddings E============================\n");


        System.out.println("\n============================information S============================\n");
        System.out.println(information);
        System.out.println("\n============================information E============================\n\n\n\n\n");

        /*
        // ***** openAI Query S *****
        String information = relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(Collectors.joining("\n\n"));

        System.out.println("Information ====> " + information);

        // [6] 프롬포트 세팅
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "\n"
                        + "Base your answer on the following information:\n"
                        + "{{information}}");

        Prompt prompt = promptTemplate.apply(variables);

        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("information", information);

        // [7] OpenAi 질의
        ChatLanguageModel chatModel = OpenAiChatModel.builder()
                .apiKey(APIKEY.OPEN_AI_KEY)
                .timeout(Duration.ofSeconds(60))
                .build();

        AiMessage aiMessage = chatModel.generate(prompt.toUserMessage()).content();
        String answer = aiMessage.text();

        System.out.println("\nAnswer is ....\n");
        System.out.println(answer);
        // ***** openAI Query E *****
        */
    }
}
