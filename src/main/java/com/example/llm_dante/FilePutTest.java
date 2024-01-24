package com.example.llm_dante;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiModelName;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import io.milvus.client.MilvusServiceClient;
import io.milvus.param.ConnectParam;
import io.milvus.param.IndexType;
import io.milvus.param.dml.InsertParam;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class FilePutTest {

    public static void main(String[] args) {

        String userDir = System.getProperty("user.dir");
        Path resourceDirectory = Paths.get(userDir, "src/main/resources/data/test2");

        // [1] 파일 로드 (폴더 업로드)
        List<Document> documents = FileSystemDocumentLoader.loadDocuments(resourceDirectory, new TextDocumentParser());

        // [2] 문장 분할
        DocumentSplitter splitter = DocumentSplitters.recursive(
                100,
                0,
                new OpenAiTokenizer(OpenAiModelName.GPT_3_5_TURBO)
        );

        List<TextSegment> segments = splitter.splitAll(documents);

        // [3] 임베딩 처리
        EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = MilvusEmbeddingStore.builder()
                .host("localhost")
                .port(19530)
                .collectionName(CONSTANT.COLLECTION_NAME)
                .dimension(384)
                .indexType(IndexType.FLAT)
                .build();

        // [4] 데이터 Insert

        // embeddingStore.addAll(embeddings, segments); // field를 직접 설정하면 아래와 같이 필드에 맞게 데이터를 넣어줘야 햠.

        // data를 넣을 client 연결
        final MilvusServiceClient milvusClient = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost("localhost")
                        .withPort(19530)
                        .build()
        );

        // 각 field에 넣을 데이터 set( id[PK]는 최초에 자동할당으로 해둠. txtField는 text segments 값, vectorFiled는 임베딩한 segments 값 )
        List<InsertParam.Field> fields = new ArrayList<>();
        fields.add(new InsertParam.Field(CONSTANT.TEXT_FIELD, segments.stream()
                .map(TextSegment::text)
                .collect(toList())));
        fields.add(new InsertParam.Field(CONSTANT.VECTOR_FIELD, embeddings.stream()
                .map(Embedding::vectorAsList)
                .collect(toList())));

        // client로 데이터 insert 수행
        milvusClient.insert(
            InsertParam.newBuilder()
                    .withCollectionName(CONSTANT.COLLECTION_NAME)
                    .withFields(fields)
                    .build()
        );
    }
}
