package com.example.llm_dante;

import io.milvus.client.MilvusServiceClient;
import io.milvus.param.ConnectParam;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MilvusTest {

    public static void main(String[] args) {
        SpringApplication.run(MilvusTest.class, args);
    }

}
